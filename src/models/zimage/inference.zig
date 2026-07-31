const std = @import("std");
const zml = @import("zml");
const zigimg = @import("zigimg");

const common = @import("../common.zig");
const zimage_model = @import("model.zig");
const zimage_text_encoder = @import("text_encoder.zig");
const zimage_tokenizer = @import("tokenizer.zig");
const zimage_scheduler = @import("scheduler.zig");
const zimage_transformer = @import("transformer.zig");
const zimage_vae = @import("vae.zig");

const log = std.log.scoped(.zimage);

fn ensurePngOutputPath(allocator: std.mem.Allocator, output_path: []const u8) ![]const u8 {
    if (std.mem.endsWith(u8, output_path, ".png")) {
        return allocator.dupe(u8, output_path);
    }

    const ext = std.fs.path.extension(output_path);
    if (ext.len > 0) {
        const stem = output_path[0 .. output_path.len - ext.len];
        return std.fmt.allocPrint(allocator, "{s}.png", .{stem});
    }

    return std.fmt.allocPrint(allocator, "{s}.png", .{output_path});
}

pub const InferenceErrors = error{
    MissingPrompt,
    InvalidImageSize,
    PromptTooLong,
    InvalidDecodedImage,
};

pub const RunOptions = struct {
    prompt: []const u8,
    negative_prompt: []const u8 = "",
    height: u32 = 1024,
    width: u32 = 1024,
    num_inference_steps: u32 = 50,
    guidance_scale: f32 = 5.0,
    cfg_normalization: bool = false,
    output_path: []const u8,
};

pub const CompilationParameters = struct {
    prompt_seqlen: u32,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    shardings: common.Shardings,

    pub fn init(
        prompt_seqlen: u32,
        latent_frames: u32,
        latent_height: u32,
        latent_width: u32,
        shardings: common.Shardings,
    ) CompilationParameters {
        return .{
            .prompt_seqlen = prompt_seqlen,
            .latent_frames = latent_frames,
            .latent_height = latent_height,
            .latent_width = latent_width,
            .shardings = shardings,
        };
    }
};

pub const CompiledModel = struct {
    loaded_model: *const zimage_model.LoadedModel,
    params: CompilationParameters,

    text_encoder: zml.Exe,
    transformer: zml.Exe,
    cfg_transformer: zml.Exe,
    latent_generator: zml.Exe,
    scheduler_step: zml.Exe,
    guided_scheduler_step: zml.Exe,
    vae_decode: zml.Exe,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded_model: *const zimage_model.LoadedModel,
        params: CompilationParameters,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const compiled_result = try compileExecutables(allocator, io, platform, loaded_model, params, progress);
        return .{
            .loaded_model = loaded_model,
            .params = params,

            .text_encoder = compiled_result.text_encoder,
            .transformer = compiled_result.transformer,
            .cfg_transformer = compiled_result.cfg_transformer,
            .latent_generator = compiled_result.latent_generator,
            .scheduler_step = compiled_result.scheduler_step,
            .guided_scheduler_step = compiled_result.guided_scheduler_step,
            .vae_decode = compiled_result.vae_decode,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.text_encoder.deinit();
        self.transformer.deinit();
        self.cfg_transformer.deinit();
        self.latent_generator.deinit();
        self.scheduler_step.deinit();
        self.guided_scheduler_step.deinit();
        self.vae_decode.deinit();
    }
};

pub const CompiledModelResult = struct {
    text_encoder: zml.Exe,
    transformer: zml.Exe,
    cfg_transformer: zml.Exe,
    latent_generator: zml.Exe,
    scheduler_step: zml.Exe,
    guided_scheduler_step: zml.Exe,
    vae_decode: zml.Exe,
};

const F32Buffer = struct {
    buffer: zml.Buffer,
    owned: bool,
};

const PromptInputs = struct {
    ids: zml.Buffer,
    mask: zml.Buffer,

    fn deinit(self: *PromptInputs) void {
        self.ids.deinit();
        self.mask.deinit();
    }
};

const EncodedPrompt = struct {
    embeds: zml.Buffer,
    token_count: zml.Buffer,
    aligned_length: zml.Buffer,

    fn deinit(self: *EncodedPrompt) void {
        self.embeds.deinit();
        self.token_count.deinit();
        self.aligned_length.deinit();
    }
};

const EncodedPrompts = struct {
    positive: EncodedPrompt,
    negative: ?EncodedPrompt,

    fn deinit(self: *EncodedPrompts) void {
        self.positive.deinit();
        if (self.negative) |*negative| negative.deinit();
    }
};

const CfgTransformerOutput = struct {
    positive: zml.Buffer,
    negative: zml.Buffer,

    fn deinit(self: *CfgTransformerOutput) void {
        self.positive.deinit();
        self.negative.deinit();
    }
};

const ExecutionState = struct {
    args: zml.Exe.Arguments,
    results: zml.Exe.Results,

    fn init(allocator: std.mem.Allocator, exe: *const zml.Exe) !ExecutionState {
        var args = try exe.args(allocator);
        errdefer args.deinit(allocator);
        const results = try exe.results(allocator);
        return .{ .args = args, .results = results };
    }

    fn deinit(self: *ExecutionState, allocator: std.mem.Allocator) void {
        self.args.deinit(allocator);
        self.results.deinit(allocator);
    }
};

const ExecutionRunner = struct {
    text_encoder: ExecutionState,
    transformer: ExecutionState,
    cfg_transformer: ExecutionState,
    latent_generator: ExecutionState,
    scheduler_step: ExecutionState,
    guided_scheduler_step: ExecutionState,
    vae_decode: ExecutionState,

    fn init(
        allocator: std.mem.Allocator,
        compiled_model: *const CompiledModel,
        model_buffers: *zimage_model.Buffers,
    ) !ExecutionRunner {
        var text_encoder = try ExecutionState.init(allocator, &compiled_model.text_encoder);
        errdefer text_encoder.deinit(allocator);
        text_encoder.args.bake(&model_buffers.text_encoder.inner);

        var transformer = try ExecutionState.init(allocator, &compiled_model.transformer);
        errdefer transformer.deinit(allocator);
        transformer.args.bake(&model_buffers.transformer);

        var cfg_transformer = try ExecutionState.init(allocator, &compiled_model.cfg_transformer);
        errdefer cfg_transformer.deinit(allocator);
        cfg_transformer.args.bake(&model_buffers.transformer);

        var latent_generator = try ExecutionState.init(allocator, &compiled_model.latent_generator);
        errdefer latent_generator.deinit(allocator);

        var scheduler_step = try ExecutionState.init(allocator, &compiled_model.scheduler_step);
        errdefer scheduler_step.deinit(allocator);

        var guided_scheduler_step = try ExecutionState.init(allocator, &compiled_model.guided_scheduler_step);
        errdefer guided_scheduler_step.deinit(allocator);

        var vae_decode = try ExecutionState.init(allocator, &compiled_model.vae_decode);
        errdefer vae_decode.deinit(allocator);
        vae_decode.args.bake(&model_buffers.vae);

        return .{
            .text_encoder = text_encoder,
            .transformer = transformer,
            .cfg_transformer = cfg_transformer,
            .latent_generator = latent_generator,
            .scheduler_step = scheduler_step,
            .guided_scheduler_step = guided_scheduler_step,
            .vae_decode = vae_decode,
        };
    }

    fn deinit(self: *ExecutionRunner, allocator: std.mem.Allocator) void {
        self.text_encoder.deinit(allocator);
        self.transformer.deinit(allocator);
        self.cfg_transformer.deinit(allocator);
        self.latent_generator.deinit(allocator);
        self.scheduler_step.deinit(allocator);
        self.guided_scheduler_step.deinit(allocator);
        self.vae_decode.deinit(allocator);
    }
};

const StepDeviceInputs = struct {
    timestep: zml.Buffer,
    current_sigma: zml.Buffer,
    next_sigma: zml.Buffer,

    fn deinit(self: *StepDeviceInputs) void {
        self.timestep.deinit();
        self.current_sigma.deinit();
        self.next_sigma.deinit();
    }
};

const DenoisingDeviceInputs = struct {
    steps: []StepDeviceInputs,
    guidance_scale: ?zml.Buffer,
    cfg_normalization: ?zml.Buffer,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        scheduler_state: *const zimage_scheduler.Scheduler,
        guidance_scale: ?f32,
        cfg_normalization: bool,
    ) !DenoisingDeviceInputs {
        std.debug.assert(scheduler_state.sigmas.len == scheduler_state.timesteps.len + 1);

        const steps = try allocator.alloc(StepDeviceInputs, scheduler_state.timesteps.len);
        errdefer allocator.free(steps);

        var initialized_steps: usize = 0;
        errdefer for (steps[0..initialized_steps]) |*step| step.deinit();

        for (steps, scheduler_state.timesteps, 0..) |*step, raw_timestep, i| {
            const normalized_timestep = (1000.0 - raw_timestep) / 1000.0;
            const timestep_values = [_]f32{normalized_timestep};
            var timestep = try zml.Buffer.fromBytes(
                io,
                platform,
                zml.Shape.init(.{ .b = 1 }, .f32),
                .replicated,
                std.mem.sliceAsBytes(&timestep_values),
            );
            errdefer timestep.deinit();

            var current_sigma = try zml.Buffer.scalar(io, platform, scheduler_state.sigmas[i], .f32);
            errdefer current_sigma.deinit();
            const next_sigma = try zml.Buffer.scalar(io, platform, scheduler_state.sigmas[i + 1], .f32);

            step.* = .{
                .timestep = timestep,
                .current_sigma = current_sigma,
                .next_sigma = next_sigma,
            };
            initialized_steps = i + 1;
        }

        var guidance_scale_buffer: ?zml.Buffer = null;
        errdefer if (guidance_scale_buffer) |*buffer| buffer.deinit();
        var cfg_normalization_buffer: ?zml.Buffer = null;
        errdefer if (cfg_normalization_buffer) |*buffer| buffer.deinit();

        if (guidance_scale) |scale| {
            guidance_scale_buffer = try zml.Buffer.scalar(io, platform, scale, .f32);
            cfg_normalization_buffer = try zml.Buffer.scalar(io, platform, cfg_normalization, .bool);
        }

        return .{
            .steps = steps,
            .guidance_scale = guidance_scale_buffer,
            .cfg_normalization = cfg_normalization_buffer,
        };
    }

    fn deinit(self: *DenoisingDeviceInputs, allocator: std.mem.Allocator) void {
        for (self.steps) |*step| step.deinit();
        allocator.free(self.steps);
        if (self.guidance_scale) |*buffer| buffer.deinit();
        if (self.cfg_normalization) |*buffer| buffer.deinit();
    }
};

fn bufferToF32(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    buffer: zml.Buffer,
) !F32Buffer {
    if (buffer.shape().dtype() == .f32) {
        return .{
            .buffer = buffer,
            .owned = false,
        };
    }

    var src = try buffer.toSliceAlloc(allocator, io);
    defer src.free(allocator);

    var dst = try zml.Slice.alloc(allocator, buffer.shape().withDtype(.f32));
    errdefer dst.free(allocator);

    switch (src.dtype()) {
        inline else => |dt| {
            const src_items = src.constItems(dt.toZigType());
            const dst_items = dst.items(f32);
            for (src_items, dst_items) |value, *out| {
                out.* = switch (comptime dt.class()) {
                    .float => switch (dt) {
                        else => zml.floats.floatCast(f32, value),
                    },
                    .integer => @floatFromInt(value),
                    .bool => if (value) 1.0 else 0.0,
                    else => unreachable,
                };
            }
        },
    }

    const converted = try zml.Buffer.fromSlice(io, platform, dst, .replicated);
    dst.free(allocator);
    return .{
        .buffer = converted,
        .owned = true,
    };
}

fn compileTextEncoderExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    text_encoder: *const zimage_text_encoder.Qwen3ForCausalLM,
    prompt_seqlen: u32,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const input_ids: zml.Tensor = .init(.{ .b = 1, .s = prompt_seqlen }, .u32);
    const attention_mask: zml.Tensor = .init(.{ .b = 1, .s = prompt_seqlen }, .bool);
    return platform.compileModel(
        allocator,
        io,
        zimage_text_encoder.Qwen3ForCausalLM.encodePrompt,
        text_encoder.*,
        .{ input_ids, attention_mask },
        .{ .shardings = shardings },
    );
}

fn compileTransformerExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    transformer: *const zimage_transformer.Transformer,
    prompt_seqlen: u32,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    hidden_size: u32,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const latent: zml.Tensor = .init(.{
        .c = transformer.in_channels,
        .f = latent_frames,
        .h = latent_height,
        .w = latent_width,
    }, .f32);
    const timestep: zml.Tensor = .init(.{ .b = 1 }, .f32);
    const prompt_embeds: zml.Tensor = .init(.{
        .s = prompt_seqlen,
        .d = hidden_size,
    }, .f32);
    const prompt_token_count: zml.Tensor = .init(.{}, .u32);
    const prompt_aligned_length: zml.Tensor = .init(.{}, .u32);

    return platform.compileFn(
        allocator,
        io,
        zimage_transformer.Transformer.denoiseStep,
        .{ transformer, latent, timestep, prompt_embeds, prompt_token_count, prompt_aligned_length },
        .{ .shardings = shardings },
    );
}

fn compileCfgTransformerExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    transformer: *const zimage_transformer.Transformer,
    prompt_seqlen: u32,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    hidden_size: u32,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const latent: zml.Tensor = .init(.{
        .c = transformer.in_channels,
        .f = latent_frames,
        .h = latent_height,
        .w = latent_width,
    }, .f32);
    const timestep: zml.Tensor = .init(.{ .b = 1 }, .f32);
    const prompt_shape = zml.Shape.init(.{
        .s = prompt_seqlen,
        .d = hidden_size,
    }, .f32);
    const positive_prompt_embeds: zml.Tensor = .fromShape(prompt_shape);
    const negative_prompt_embeds: zml.Tensor = .fromShape(prompt_shape);
    const length_shape = zml.Shape.init(.{}, .u32);
    const positive_token_count: zml.Tensor = .fromShape(length_shape);
    const positive_aligned_length: zml.Tensor = .fromShape(length_shape);
    const negative_token_count: zml.Tensor = .fromShape(length_shape);
    const negative_aligned_length: zml.Tensor = .fromShape(length_shape);

    return platform.compileFn(
        allocator,
        io,
        zimage_transformer.Transformer.denoiseCfgStep,
        .{
            transformer,
            latent,
            timestep,
            positive_prompt_embeds,
            positive_token_count,
            positive_aligned_length,
            negative_prompt_embeds,
            negative_token_count,
            negative_aligned_length,
        },
        .{ .shardings = shardings },
    );
}

fn generateLatents(latent_shape: zml.Shape) zml.Tensor {
    return zml.Tensor.Rng.normal(latent_shape, .{});
}

fn compileLatentGeneratorExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    latent_channels: u32,
    params: CompilationParameters,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const latent_shape = zml.Shape.init(.{
        .c = latent_channels,
        .f = params.latent_frames,
        .h = params.latent_height,
        .w = params.latent_width,
    }, .f32);

    return platform.compileFn(
        allocator,
        io,
        generateLatents,
        .{latent_shape},
        .{ .shardings = shardings },
    );
}

fn uploadPromptInputs(
    io: std.Io,
    platform: *const zml.Platform,
    encoding: *const zimage_tokenizer.PromptEncoding,
) !PromptInputs {
    var ids = try zml.Buffer.fromSlice(io, platform, encoding.ids, .replicated);
    errdefer ids.deinit();
    const mask = try zml.Buffer.fromSlice(io, platform, encoding.mask, .replicated);

    return .{
        .ids = ids,
        .mask = mask,
    };
}

fn compileSchedulerStepExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    latent_channels: u32,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    model_output_dtype: zml.DataType,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const model_output: zml.Tensor = .init(.{
        .c = latent_channels,
        .f = latent_frames,
        .h = latent_height,
        .w = latent_width,
    }, model_output_dtype);
    const sample: zml.Tensor = .init(.{
        .c = latent_channels,
        .f = latent_frames,
        .h = latent_height,
        .w = latent_width,
    }, .f32);
    const current_sigma: zml.Tensor = .init(.{}, .f32);
    const next_sigma: zml.Tensor = .init(.{}, .f32);
    const kernel: zimage_model.SchedulerStepKernel = .{};

    return platform.compile(
        allocator,
        io,
        kernel,
        .step,
        .{ model_output, sample, current_sigma, next_sigma },
        .{ .shardings = shardings },
    );
}

fn compileGuidedSchedulerStepExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    latent_channels: u32,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    model_output_dtype: zml.DataType,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const model_output_shape = zml.Shape.init(.{
        .c = latent_channels,
        .f = latent_frames,
        .h = latent_height,
        .w = latent_width,
    }, model_output_dtype);
    const sample_shape = model_output_shape.withDtype(.f32);
    const positive_output: zml.Tensor = .fromShape(model_output_shape);
    const negative_output: zml.Tensor = .fromShape(model_output_shape);
    const sample: zml.Tensor = .fromShape(sample_shape);
    const current_sigma: zml.Tensor = .init(.{}, .f32);
    const next_sigma: zml.Tensor = .init(.{}, .f32);
    const guidance_scale: zml.Tensor = .init(.{}, .f32);
    const normalize: zml.Tensor = .init(.{}, .bool);
    const kernel: zimage_model.GuidedSchedulerStepKernel = .{};

    return platform.compile(
        allocator,
        io,
        kernel,
        .step,
        .{
            positive_output,
            negative_output,
            sample,
            current_sigma,
            next_sigma,
            guidance_scale,
            normalize,
        },
        .{ .shardings = shardings },
    );
}

fn compileVaeDecodeExe(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    vae: *const zimage_vae.AutoEncoder,
    latent_frames: u32,
    latent_height: u32,
    latent_width: u32,
    shardings: []const zml.Sharding,
) !zml.Exe {
    const latents: zml.Tensor = .init(.{
        .c = vae.config.latent_channels,
        .f = latent_frames,
        .h = latent_height,
        .w = latent_width,
    }, .f32);

    return platform.compileModel(
        allocator,
        io,
        zimage_vae.AutoEncoder.decodeLatents,
        vae,
        .{latents},
        .{ .shardings = shardings },
    );
}

fn compileExecutables(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    loaded_model: *const zimage_model.LoadedModel,
    params: CompilationParameters,
    progress: *std.Progress.Node,
) !CompiledModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled executables [{f}]", .{now.untilNow(io, .awake)});

    const all_shardings = params.shardings.all();
    const text_encoder = &loaded_model.inner.text_encoder.inner;
    const transformer = &loaded_model.inner.transformer;
    const vae = &loaded_model.inner.vae;
    progress.increaseEstimatedTotalItems(7);

    var text_encoder_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            text_encoder_: *const zimage_text_encoder.Qwen3ForCausalLM,
            prompt_seqlen_: u32,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            var node = progress_.start("Compiling text encoder...", 1);
            defer node.end();

            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled text encoder [{f}]", .{now_.untilNow(io_, .awake)});

            return compileTextEncoderExe(allocator_, io_, platform_, text_encoder_, prompt_seqlen_, shardings_);
        }
    }.call, .{ allocator, io, platform, text_encoder, params.prompt_seqlen, &all_shardings, progress });
    var text_encoder_future_awaited = false;
    errdefer if (!text_encoder_future_awaited) if (text_encoder_future.cancel(io)) |exe| exe.deinit() else |_| {};

    var transformer_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            transformer_: *const zimage_transformer.Transformer,
            params_: CompilationParameters,
            hidden_size_: u32,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            var node = progress_.start("Compiling transformer...", 1);
            defer node.end();

            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled transformer [{f}]", .{now_.untilNow(io_, .awake)});

            return compileTransformerExe(
                allocator_,
                io_,
                platform_,
                transformer_,
                params_.prompt_seqlen,
                params_.latent_frames,
                params_.latent_height,
                params_.latent_width,
                hidden_size_,
                shardings_,
            );
        }
    }.call, .{ allocator, io, platform, transformer, params, loaded_model.inner.text_encoder.config.hidden_size, &all_shardings, progress });
    var transformer_future_awaited = false;
    errdefer if (!transformer_future_awaited) if (transformer_future.cancel(io)) |exe| exe.deinit() else |_| {};

    var cfg_transformer_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            transformer_: *const zimage_transformer.Transformer,
            params_: CompilationParameters,
            hidden_size_: u32,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            var node = progress_.start("Compiling CFG transformer...", 1);
            defer node.end();

            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled CFG transformer [{f}]", .{now_.untilNow(io_, .awake)});

            return compileCfgTransformerExe(
                allocator_,
                io_,
                platform_,
                transformer_,
                params_.prompt_seqlen,
                params_.latent_frames,
                params_.latent_height,
                params_.latent_width,
                hidden_size_,
                shardings_,
            );
        }
    }.call, .{ allocator, io, platform, transformer, params, loaded_model.inner.text_encoder.config.hidden_size, &all_shardings, progress });
    var cfg_transformer_future_awaited = false;
    errdefer if (!cfg_transformer_future_awaited) if (cfg_transformer_future.cancel(io)) |exe| exe.deinit() else |_| {};

    var latent_generator_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            latent_channels_: u32,
            params_: CompilationParameters,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            var node = progress_.start("Compiling latent generator...", 1);
            defer node.end();

            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled latent generator [{f}]", .{now_.untilNow(io_, .awake)});

            return compileLatentGeneratorExe(
                allocator_,
                io_,
                platform_,
                latent_channels_,
                params_,
                shardings_,
            );
        }
    }.call, .{ allocator, io, platform, transformer.in_channels, params, &all_shardings, progress });
    var latent_generator_future_awaited = false;
    errdefer if (!latent_generator_future_awaited) if (latent_generator_future.cancel(io)) |exe| exe.deinit() else |_| {};

    var scheduler_step_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            latent_channels_: u32,
            params_: CompilationParameters,
            model_output_dtype_: zml.DataType,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            var node = progress_.start("Compiling scheduler step...", 1);
            defer node.end();

            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled scheduler step [{f}]", .{now_.untilNow(io_, .awake)});

            return compileSchedulerStepExe(
                allocator_,
                io_,
                platform_,
                latent_channels_,
                params_.latent_frames,
                params_.latent_height,
                params_.latent_width,
                model_output_dtype_,
                shardings_,
            );
        }
    }.call, .{ allocator, io, platform, vae.config.latent_channels, params, transformer.all_final_layer[0].linear.weight.dtype(), &all_shardings, progress });
    var scheduler_step_future_awaited = false;
    errdefer if (!scheduler_step_future_awaited) if (scheduler_step_future.cancel(io)) |exe| exe.deinit() else |_| {};

    var guided_scheduler_step_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            latent_channels_: u32,
            params_: CompilationParameters,
            model_output_dtype_: zml.DataType,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            var node = progress_.start("Compiling guided scheduler step...", 1);
            defer node.end();

            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled guided scheduler step [{f}]", .{now_.untilNow(io_, .awake)});

            return compileGuidedSchedulerStepExe(
                allocator_,
                io_,
                platform_,
                latent_channels_,
                params_.latent_frames,
                params_.latent_height,
                params_.latent_width,
                model_output_dtype_,
                shardings_,
            );
        }
    }.call, .{ allocator, io, platform, vae.config.latent_channels, params, transformer.all_final_layer[0].linear.weight.dtype(), &all_shardings, progress });
    var guided_scheduler_step_future_awaited = false;
    errdefer if (!guided_scheduler_step_future_awaited) if (guided_scheduler_step_future.cancel(io)) |exe| exe.deinit() else |_| {};

    var vae_decode_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            vae_: *const zimage_vae.AutoEncoder,
            params_: CompilationParameters,
            shardings_: []const zml.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            var node = progress_.start("Compiling VAE decode...", 1);
            defer node.end();

            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled VAE decode [{f}]", .{now_.untilNow(io_, .awake)});

            return compileVaeDecodeExe(
                allocator_,
                io_,
                platform_,
                vae_,
                params_.latent_frames,
                params_.latent_height,
                params_.latent_width,
                shardings_,
            );
        }
    }.call, .{ allocator, io, platform, vae, params, &all_shardings, progress });
    var vae_decode_future_awaited = false;
    errdefer if (!vae_decode_future_awaited) if (vae_decode_future.cancel(io)) |exe| exe.deinit() else |_| {};

    const text_encoder_exe = try text_encoder_future.await(io);
    text_encoder_future_awaited = true;
    errdefer text_encoder_exe.deinit();

    const transformer_exe = try transformer_future.await(io);
    transformer_future_awaited = true;
    errdefer transformer_exe.deinit();

    const cfg_transformer_exe = try cfg_transformer_future.await(io);
    cfg_transformer_future_awaited = true;
    errdefer cfg_transformer_exe.deinit();

    const latent_generator_exe = try latent_generator_future.await(io);
    latent_generator_future_awaited = true;
    errdefer latent_generator_exe.deinit();

    const scheduler_step_exe = try scheduler_step_future.await(io);
    scheduler_step_future_awaited = true;
    errdefer scheduler_step_exe.deinit();

    const guided_scheduler_step_exe = try guided_scheduler_step_future.await(io);
    guided_scheduler_step_future_awaited = true;
    errdefer guided_scheduler_step_exe.deinit();

    const vae_decode_exe = try vae_decode_future.await(io);
    vae_decode_future_awaited = true;

    return .{
        .text_encoder = text_encoder_exe,
        .transformer = transformer_exe,
        .cfg_transformer = cfg_transformer_exe,
        .latent_generator = latent_generator_exe,
        .scheduler_step = scheduler_step_exe,
        .guided_scheduler_step = guided_scheduler_step_exe,
        .vae_decode = vae_decode_exe,
    };
}

pub const InferencePipeline = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled_model: CompiledModel,
    model_buffers: zimage_model.Buffers,
    text_tokenizer: zimage_tokenizer.Tokenizer,
    scheduler_state: zimage_scheduler.Scheduler,
    runner: ExecutionRunner,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        platform: *const zml.Platform,
        loaded_model: *const zimage_model.LoadedModel,
        store: *zml.io.TensorStore,
        shardings: common.Shardings,
        prompt_seqlen: u32,
        height: u32,
        width: u32,
        progress: *std.Progress.Node,
    ) !InferencePipeline {
        const backend = zml.attention.Backend.auto(platform);

        // Compile model and generate its exe's
        var compiled_model = try loaded_model.compile(
            allocator,
            io,
            platform,
            backend,
            shardings,
            prompt_seqlen,
            height,
            width,
            progress,
        );
        errdefer compiled_model.deinit();

        // Load model buffers after the model compilation
        var model_buffers = try loaded_model.loadBuffers(
            allocator,
            io,
            platform,
            store,
            progress,
            shardings,
        );
        errdefer loaded_model.unloadBuffers(&model_buffers, allocator);

        var text_tokenizer = try zimage_tokenizer.Tokenizer.fromDir(
            allocator,
            io,
            repo,
            common.repositoryFiles(.zimage),
            .{},
        );
        errdefer text_tokenizer.deinit();

        var scheduler_state = try zimage_scheduler.Scheduler.init(
            allocator,
            loaded_model.scheduler_config.value,
        );
        errdefer scheduler_state.deinit(allocator);

        var runner = try ExecutionRunner.init(allocator, &compiled_model, &model_buffers);
        errdefer runner.deinit(allocator);

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .compiled_model = compiled_model,
            .model_buffers = model_buffers,
            .text_tokenizer = text_tokenizer,
            .scheduler_state = scheduler_state,
            .runner = runner,
        };
    }

    pub fn deinit(self: *InferencePipeline, allocator: std.mem.Allocator) void {
        self.runner.deinit(allocator);
        self.compiled_model.loaded_model.unloadBuffers(&self.model_buffers, allocator);
        self.compiled_model.deinit();
        self.text_tokenizer.deinit();
        self.scheduler_state.deinit(allocator);
    }

    fn encodePrompt(
        self: *InferencePipeline,
        prompt: []const u8,
        negative_prompt: []const u8,
        do_classifier_free_guidance: bool,
    ) !EncodedPrompts {
        var positive = try self.encodePromptItem(prompt);
        errdefer positive.deinit();

        const negative = if (do_classifier_free_guidance)
            try self.encodePromptItem(negative_prompt)
        else
            null;

        return .{
            .positive = positive,
            .negative = negative,
        };
    }

    fn encodePromptItem(
        self: *InferencePipeline,
        prompt: []const u8,
    ) !EncodedPrompt {
        var prompt_encoding = try self.text_tokenizer.encodePromptAlloc(
            self.allocator,
            prompt,
            self.compiled_model.params.prompt_seqlen,
        );
        defer prompt_encoding.deinit(self.allocator);

        var prompt_inputs = try uploadPromptInputs(self.io, self.platform, &prompt_encoding);
        defer prompt_inputs.deinit();

        self.runner.text_encoder.args.set(.{
            prompt_inputs.ids,
            prompt_inputs.mask,
        });
        self.compiled_model.text_encoder.call(self.runner.text_encoder.args, &self.runner.text_encoder.results);

        // Diffusers strips Qwen padding, then pads each caption to a multiple
        // of 32 inside the transformer. Keep the fixed-size embeddings and
        // carry the two runtime lengths needed to reproduce those semantics.
        var embeds = self.runner.text_encoder.results.get(zml.Buffer);
        errdefer embeds.deinit();
        var token_count = try zml.Buffer.scalar(
            self.io,
            self.platform,
            prompt_encoding.token_count,
            .u32,
        );
        errdefer token_count.deinit();
        const aligned_token_count = std.mem.alignForward(u32, prompt_encoding.token_count, zimage_transformer.SEQ_MULTI_OF);
        const aligned_length = try zml.Buffer.scalar(
            self.io,
            self.platform,
            aligned_token_count,
            .u32,
        );

        var prompt_embeds: EncodedPrompt = .{
            .embeds = embeds,
            .token_count = token_count,
            .aligned_length = aligned_length,
        };
        errdefer prompt_embeds.deinit();

        return prompt_embeds;
    }

    pub fn run(self: *InferencePipeline, opts: RunOptions) !void {
        if (opts.prompt.len == 0) return InferenceErrors.MissingPrompt;
        const do_cfg = opts.guidance_scale > 0.0;

        if (opts.height % 16 != 0 or opts.width % 16 != 0) return InferenceErrors.InvalidImageSize;

        const latent_height = opts.height / 8;
        const latent_width = opts.width / 8;
        if (latent_height != self.compiled_model.params.latent_height or latent_width != self.compiled_model.params.latent_width) {
            return InferenceErrors.InvalidImageSize;
        }

        var prompt_embeds = try self.encodePrompt(
            opts.prompt,
            if (opts.negative_prompt.len == 0) "" else opts.negative_prompt,
            do_cfg,
        );
        defer prompt_embeds.deinit();

        // Generate latents
        self.compiled_model.latent_generator.call(self.runner.latent_generator.args, &self.runner.latent_generator.results);
        var latents = self.runner.latent_generator.results.get(zml.Buffer);
        defer latents.deinit();

        const image_seq_len = (self.compiled_model.params.latent_height / 2) * (self.compiled_model.params.latent_width / 2);

        // Prepare timesteps
        const mu: ?f32 = if (self.scheduler_state.use_dynamic_shifting) blk: {
            const base_seq_len = @as(f32, @floatFromInt(self.scheduler_state.base_image_seq_len));
            const max_seq_len = @as(f32, @floatFromInt(self.scheduler_state.max_image_seq_len));
            const base_shift = self.scheduler_state.base_shift orelse 0.5;
            const max_shift = self.scheduler_state.max_shift orelse 1.15;
            const slope = (max_shift - base_shift) / (max_seq_len - base_seq_len);
            const intercept = base_shift - slope * base_seq_len;
            break :blk @as(f32, @floatFromInt(image_seq_len)) * slope + intercept;
        } else null;

        // Set timesteps
        try self.scheduler_state.setTimesteps(self.allocator, opts.num_inference_steps, mu);

        var denoising_inputs = try DenoisingDeviceInputs.init(
            self.allocator,
            self.io,
            self.platform,
            &self.scheduler_state,
            if (do_cfg) opts.guidance_scale else null,
            opts.cfg_normalization,
        );
        defer denoising_inputs.deinit(self.allocator);

        if (do_cfg) {
            // Model buffers are baked. Prompt buffers and their lengths are
            // constant for every denoising step, so set them once.
            self.runner.cfg_transformer.args.setPartial(.{
                prompt_embeds.positive.embeds,
                prompt_embeds.positive.token_count,
                prompt_embeds.positive.aligned_length,
                prompt_embeds.negative.?.embeds,
                prompt_embeds.negative.?.token_count,
                prompt_embeds.negative.?.aligned_length,
            }, 2);
        } else {
            self.runner.transformer.args.setPartial(.{
                prompt_embeds.positive.embeds,
                prompt_embeds.positive.token_count,
                prompt_embeds.positive.aligned_length,
            }, 2);
        }

        // Denoising loop
        for (denoising_inputs.steps, 0..) |*step_inputs, step_index| {
            const step_number = step_index + 1;
            const step_started: std.Io.Timestamp = .now(self.io, .awake);

            const next_latents = if (do_cfg) blk: {
                self.runner.cfg_transformer.args.set(.{ latents, step_inputs.timestep });
                self.compiled_model.cfg_transformer.call(
                    self.runner.cfg_transformer.args,
                    &self.runner.cfg_transformer.results,
                );

                var cfg_output = self.runner.cfg_transformer.results.get(CfgTransformerOutput);
                defer cfg_output.deinit();

                break :blk try self.runGuidedSchedulerStep(
                    cfg_output.positive,
                    cfg_output.negative,
                    latents,
                    step_inputs,
                    &denoising_inputs,
                );
            } else blk: {
                self.runner.transformer.args.set(.{ latents, step_inputs.timestep });
                self.compiled_model.transformer.call(
                    self.runner.transformer.args,
                    &self.runner.transformer.results,
                );

                var positive_output = self.runner.transformer.results.get(zml.Buffer);
                defer positive_output.deinit();

                break :blk try self.runSchedulerStep(positive_output, latents, step_inputs);
            };

            latents.deinit();
            latents = next_latents;
            log.info(
                "Denoising step {d}/{d} [{f}]",
                .{ step_number, denoising_inputs.steps.len, step_started.untilNow(self.io, .awake) },
            );
        }

        var scaled_latent_slice = try latents.toSliceAlloc(self.allocator, self.io);
        defer scaled_latent_slice.free(self.allocator);
        const scaling_factor = self.compiled_model.loaded_model.inner.vae.config.scaling_factor;
        const shift_factor = self.compiled_model.loaded_model.inner.vae.config.shift_factor;
        for (scaled_latent_slice.items(f32)) |*value| {
            value.* = (value.* / scaling_factor) + shift_factor;
        }

        var scaled_latents = try zml.Buffer.fromSlice(self.io, self.platform, scaled_latent_slice, .replicated);
        defer scaled_latents.deinit();

        self.runner.vae_decode.args.set(.{scaled_latents});
        self.compiled_model.vae_decode.call(self.runner.vae_decode.args, &self.runner.vae_decode.results);

        var image_buffer = self.runner.vae_decode.results.get(zml.Buffer);
        defer image_buffer.deinit();

        var image_f32 = try bufferToF32(self.allocator, self.io, self.platform, image_buffer);
        defer if (image_f32.owned) image_f32.buffer.deinit();

        var image_slice = try image_f32.buffer.toSliceAlloc(self.allocator, self.io);
        defer image_slice.free(self.allocator);

        if (image_slice.shape.rank() != 4 or image_slice.shape.dim(.b) != 1 or image_slice.shape.dim(.c) < 3) {
            return InferenceErrors.InvalidDecodedImage;
        }

        const image_height: usize = @intCast(image_slice.shape.dim(.h));
        const image_width: usize = @intCast(image_slice.shape.dim(.w));
        const image_items = image_slice.constItems(f32);

        var pixels = try self.allocator.alloc(zigimg.color.Rgb24, image_width * image_height);
        for (0..image_height) |y| {
            for (0..image_width) |x| {
                var rgb: [3]u8 = undefined;
                for (0..3) |c| {
                    const idx = ((c * image_height + y) * image_width) + x;
                    const value = std.math.clamp(image_items[idx], -1.0, 1.0) * 0.5 + 0.5;
                    rgb[c] = @intFromFloat(std.math.clamp(value * 255.0, 0.0, 255.0));
                }
                pixels[y * image_width + x] = zigimg.color.Rgb24.from.rgb(rgb[0], rgb[1], rgb[2]);
            }
        }

        var image: zigimg.Image = .{
            .width = image_width,
            .height = image_height,
            .pixels = .{ .rgb24 = pixels },
            .animation = .{},
        };
        defer image.deinit(self.allocator);

        const output_path = try ensurePngOutputPath(self.allocator, opts.output_path);
        defer self.allocator.free(output_path);

        var write_buffer: [zigimg.io.DEFAULT_BUFFER_SIZE]u8 = undefined;
        try image.writeToFilePath(self.allocator, self.io, output_path, write_buffer[0..], .{ .png = .{} });
    }

    pub fn runSchedulerStep(
        self: *InferencePipeline,
        noise_pred: zml.Buffer,
        latents: zml.Buffer,
        step_inputs: *const StepDeviceInputs,
    ) !zml.Buffer {
        self.runner.scheduler_step.args.set(.{
            noise_pred,
            latents,
            step_inputs.current_sigma,
            step_inputs.next_sigma,
        });
        self.compiled_model.scheduler_step.call(self.runner.scheduler_step.args, &self.runner.scheduler_step.results);

        return self.runner.scheduler_step.results.get(zml.Buffer);
    }

    fn runGuidedSchedulerStep(
        self: *InferencePipeline,
        positive_output: zml.Buffer,
        negative_output: zml.Buffer,
        latents: zml.Buffer,
        step_inputs: *const StepDeviceInputs,
        denoising_inputs: *const DenoisingDeviceInputs,
    ) !zml.Buffer {
        self.runner.guided_scheduler_step.args.set(.{
            positive_output,
            negative_output,
            latents,
            step_inputs.current_sigma,
            step_inputs.next_sigma,
            denoising_inputs.guidance_scale.?,
            denoising_inputs.cfg_normalization.?,
        });
        self.compiled_model.guided_scheduler_step.call(
            self.runner.guided_scheduler_step.args,
            &self.runner.guided_scheduler_step.results,
        );

        return self.runner.guided_scheduler_step.results.get(zml.Buffer);
    }
};
