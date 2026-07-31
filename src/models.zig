const std = @import("std");
const zml = @import("zml");

pub const zimage = @import("models/zimage.zig");
pub const pipeline = zimage.inference.InferencePipeline;
pub const InferenceErrors = zimage.inference.InferenceErrors;
pub const common = @import("models/common.zig");
pub const Shardings = common.Shardings;
pub const RepositoryFiles = common.RepositoryFiles;
pub const ModelType = common.ModelType;

const log = std.log.scoped(.diffusion);

const RawConfig = struct {
    _class_name: []const u8,
};

pub const LoadedModel = union(ModelType) {
    zimage: zimage.LoadedModel,

    pub fn load(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        model_type: ModelType,
    ) !LoadedModel {
        log.info("Detected mode type: {}", .{model_type});

        return switch (model_type) {
            .zimage => .{ .zimage = try zimage.LoadedModel.init(allocator, io, repo, store) },
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .zimage => |*m| {
                m.deinit(allocator);
            },
        }
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.Backend,
        shardings: Shardings,
        seqlen: usize,
        height: u32,
        width: u32,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        const inner: CompiledModel.Inner = switch (self.*) {
            .zimage => |*m| .{ .zimage = try m.compile(
                allocator,
                io,
                platform,
                backend,
                shardings,
                seqlen,
                height,
                width,
                progress,
            ) },
        };

        return .{
            .inner = inner,
            .seqlen = @intCast(seqlen),
        };
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: Shardings,
    ) !Buffers {
        return switch (self.*) {
            .zimage => |*m| .{ .zimage = try m.loadBuffers(allocator, io, platform, store, progress, shardings) },
        };
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .zimage => |*loaded_model| switch (buffers.*) {
                .zimage => |*loaded_buffers| loaded_model.unloadBuffers(loaded_buffers, allocator),
            },
        }
    }
};

pub const CompiledModel = struct {
    const Inner = union(ModelType) {
        zimage: zimage.inference.CompiledModel,
    };

    inner: Inner,
    seqlen: u32,

    pub fn deinit(self: *CompiledModel) void {
        switch (self.inner) {
            inline else => |*m| m.deinit(),
        }
    }
};

pub const Buffers = union(ModelType) {
    zimage: zimage.Buffers,
};

pub fn repositoryFiles(model_type: ModelType) RepositoryFiles {
    return common.repositoryFiles(model_type);
}

pub fn detectModelType(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir) !ModelType {
    const file = try repo.openFile(io, common.repositoryFiles(.zimage).model_index, .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();
    const parsed = try std.json.parseFromTokenSource(RawConfig, allocator, &reader, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    if (std.mem.eql(u8, parsed.value._class_name, "ZImagePipeline")) return .zimage;
    return error.UnknownModelType;
}

test "models" {
    std.testing.refAllDecls(@This());
    _ = zimage.scheduler;
    _ = zimage.text_encoder;
    _ = zimage.transformer;
}
