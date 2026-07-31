const std = @import("std");
const zml = @import("zml");

const common = @import("../common.zig");
const inference = @import("inference.zig");
const scheduler = @import("scheduler.zig");
const text_encoder = @import("text_encoder.zig");
const transformer = @import("transformer.zig");
const vae = @import("vae.zig");

const log = std.log.scoped(.@"diffusion/zimage");

pub const Config = struct {
    text_encoder: text_encoder.TextEncoder.Config,
    transformer: transformer.Transformer.Config,
    vae: vae.AutoEncoder.Config,
    scheduler: scheduler.SchedulerConfig,
};

pub const Buffers = zml.Bufferized(Model);

const load_parallelism = 2;
const load_dma_chunks = 4;
const load_dma_chunk_size = 16 * zml.MiB;

fn initializeBufferTree(
    allocator: std.mem.Allocator,
    model: anytype,
    buffers: *zml.Bufferized(@TypeOf(model.*)),
) !void {
    @setEvalBranchQuota(10_000);
    const Bufferized = zml.Bufferized(@TypeOf(model.*));

    if (Bufferized == zml.Buffer) {
        buffers._shards = .empty;
        return;
    }

    switch (@typeInfo(Bufferized)) {
        .@"struct" => |struct_info| {
            inline for (struct_info.fields) |field| {
                try initializeBufferTree(
                    allocator,
                    &@field(model.*, field.name),
                    &@field(buffers, field.name),
                );
            }
        },
        .@"union" => switch (model.*) {
            inline else => |*value, tag| {
                buffers.* = @unionInit(Bufferized, @tagName(tag), undefined);
                try initializeBufferTree(allocator, value, &@field(buffers, @tagName(tag)));
            },
        },
        .optional => |optional_info| {
            if (model.*) |*value| {
                // Wrap only after initialization so the optional has a valid present tag.
                var payload: optional_info.child = undefined;
                try initializeBufferTree(allocator, value, &payload);
                buffers.* = payload;
            } else {
                buffers.* = null;
            }
        },
        .pointer => |pointer_info| switch (pointer_info.size) {
            .slice => {
                buffers.* = try allocator.alloc(pointer_info.child, model.*.len);
                for (model.*, buffers.*) |*source, *destination| {
                    try initializeBufferTree(allocator, source, destination);
                }
            },
            else => unreachable,
        },
        .void, .int, .@"enum", .bool, .enum_literal, .float, .vector => {},
        else => unreachable,
    }
}

pub const Model = struct {
    text_encoder: text_encoder.TextEncoder,
    transformer: transformer.Transformer,
    vae: vae.AutoEncoder,

    pub fn init(
        self: *Model,
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: *const Config,
    ) !void {
        log.info("Initializing text encoder model graph...", .{});
        self.text_encoder = try text_encoder.TextEncoder.init(allocator, store.withPrefix("text_encoder"), config.text_encoder);
        errdefer self.text_encoder.deinit(allocator);

        log.info("Initializing transformer model graph...", .{});
        try self.transformer.init(allocator, store.withPrefix("transformer"), config.transformer);
        errdefer self.transformer.deinit(allocator);

        log.info("Initializing VAE model graph...", .{});
        self.vae = try vae.AutoEncoder.init(store.withPrefix("vae"), config.vae);
        errdefer self.vae.deinit(allocator);
    }

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        self.text_encoder.deinit(allocator);
        self.transformer.deinit(allocator);
        self.vae.deinit(allocator);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        text_encoder.Qwen3Model.unloadBuffers(&self.text_encoder.inner.model, allocator);
        if (self.text_encoder.inner.lm_head) |*lm_head| lm_head.weight.deinit();
        transformer.Transformer.unloadBuffers(&self.transformer, allocator);
        vae.AutoEncoder.unloadBuffers(&self.vae, allocator);
    }
};

pub const SchedulerStepKernel = struct {
    pub fn step(
        _: SchedulerStepKernel,
        model_output: zml.Tensor,
        sample: zml.Tensor,
        current_sigma: zml.Tensor,
        next_sigma: zml.Tensor,
    ) zml.Tensor {
        const sample_f32 = sample.convert(.f32);
        const dt = next_sigma.convert(.f32)
            .sub(current_sigma.convert(.f32))
            .broad(sample_f32.shape());

        return sample_f32
            .add(dt.mul(model_output.convert(.f32).negate()))
            .convert(sample.dtype());
    }
};

pub const GuidedSchedulerStepKernel = struct {
    pub fn step(
        _: GuidedSchedulerStepKernel,
        positive_output: zml.Tensor,
        negative_output: zml.Tensor,
        sample: zml.Tensor,
        current_sigma: zml.Tensor,
        next_sigma: zml.Tensor,
        guidance_scale: zml.Tensor,
        normalize: zml.Tensor,
    ) zml.Tensor {
        const positive_f32 = positive_output.convert(.f32);
        const negative_f32 = negative_output.convert(.f32);
        var guided = positive_f32.add(
            positive_f32
                .sub(negative_f32)
                .mul(guidance_scale.convert(.f32).broad(positive_f32.shape())),
        );

        const positive_norm = vectorNorm(positive_f32);
        const guided_norm = vectorNorm(guided);
        const should_normalize = normalize.convert(.bool).logical(
            .AND,
            guided_norm.cmp(.GT, positive_norm),
        );
        const normalization_scale = positive_norm
            .div(guided_norm.maximum(zml.Tensor.scalar(std.math.floatEps(f32), .f32)));
        const scale = should_normalize.select(normalization_scale, zml.Tensor.scalar(1.0, .f32));
        guided = guided.mul(scale.broad(guided.shape()));

        const sample_f32 = sample.convert(.f32);
        const dt = next_sigma.convert(.f32)
            .sub(current_sigma.convert(.f32))
            .broad(sample_f32.shape());

        return sample_f32
            .add(dt.mul(guided.negate()))
            .convert(sample.dtype());
    }

    fn vectorNorm(tensor: zml.Tensor) zml.Tensor {
        const flattened = tensor.flatten();
        return flattened
            .mul(flattened)
            .sum(0)
            .sqrt()
            .squeeze(0);
    }
};

pub const LoadedModel = struct {
    inner: Model,
    scheduler_state: scheduler.Scheduler,
    text_encoder_config: std.json.Parsed(text_encoder.TextEncoder.Config),
    transformer_config: std.json.Parsed(transformer.Transformer.Config),
    vae_config: std.json.Parsed(vae.AutoEncoder.Config),
    scheduler_config: std.json.Parsed(scheduler.SchedulerConfig),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
    ) !LoadedModel {
        const files = common.repositoryFiles(.zimage);

        log.info("Parsing text encoder config...", .{});
        const text_encoder_config = try common.parse_config_at_path(
            text_encoder.TextEncoder.Config,
            allocator,
            io,
            repo,
            files.text_encoder_config,
        );
        errdefer text_encoder_config.deinit();
        log.info("Parsing transformer config...", .{});
        const transformer_config = try common.parse_config_at_path(
            transformer.Transformer.Config,
            allocator,
            io,
            repo,
            files.transformer_config,
        );
        errdefer transformer_config.deinit();
        log.info("Parsing VAE config...", .{});
        const vae_config = try common.parse_config_at_path(
            vae.AutoEncoder.Config,
            allocator,
            io,
            repo,
            files.vae_config,
        );
        errdefer vae_config.deinit();
        log.info("Parsing scheduler config...", .{});
        const scheduler_config = try parseSchedulerConfig(allocator, io, repo, files.scheduler_config);
        errdefer scheduler_config.deinit();

        log.info("Building merged Z-Image config...", .{});
        const merged_config: Config = .{
            .text_encoder = text_encoder_config.value,
            .transformer = transformer_config.value,
            .vae = vae_config.value,
            .scheduler = scheduler_config.value,
        };

        log.info("Initializing merged Z-Image model...", .{});
        var inner: Model = undefined;
        try inner.init(allocator, store, &merged_config);

        return .{
            .inner = inner,
            .scheduler_state = try .init(allocator, scheduler_config.value),
            .text_encoder_config = text_encoder_config,
            .transformer_config = transformer_config,
            .vae_config = vae_config,
            .scheduler_config = scheduler_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.scheduler_state.deinit(allocator);
        self.text_encoder_config.deinit();
        self.transformer_config.deinit();
        self.vae_config.deinit();
        self.scheduler_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        progress.increaseEstimatedTotalItems(store.view().count());

        var buffers: Buffers = undefined;
        try initializeBufferTree(allocator, &self.inner, &buffers);
        errdefer Model.unloadBuffers(&buffers, allocator);

        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = load_dma_chunks,
            .dma_chunk_size = load_dma_chunk_size,
            .parallelism = load_parallelism,
        });
        defer loader.deinit();

        const all_shardings = shardings.all();
        loader.load(io, Model, &self.inner, &buffers, store, &all_shardings, .{ .progress = progress });
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self;
        Model.unloadBuffers(buffers, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.Backend,
        shardings: common.Shardings,
        seqlen: u32,
        height: u32,
        width: u32,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = backend;
        const params = inference.CompilationParameters.init(
            seqlen,
            1, // Latent frame
            height / 8,
            width / 8,
            shardings,
        );
        return inference.CompiledModel.init(allocator, io, platform, self, params, progress);
    }
};

fn parseSchedulerConfig(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo: std.Io.Dir,
    path: []const u8,
) !std.json.Parsed(scheduler.SchedulerConfig) {
    const file = try repo.openFile(io, path, .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(scheduler.SchedulerConfig, allocator, &reader, .{ .ignore_unknown_fields = true });
}
