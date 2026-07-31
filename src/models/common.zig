const std = @import("std");
const zml = @import("zml");

pub const ModelType = enum {
    zimage,
};

pub const RepositoryFiles = struct {
    model_index: []const u8,
    tensor_entrypoints: []const []const u8,
    text_encoder_config: []const u8,
    transformer_config: []const u8,
    vae_config: []const u8,
    scheduler_config: []const u8,
    tokenizer: []const []const u8,
    tokenizer_config: []const []const u8,
};

const zimage_tensor_entrypoints = [_][]const u8{
    "text_encoder/model.safetensors.index.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/diffusion_pytorch_model.safetensors",
};

const zimage_tokenizer_files = [_][]const u8{
    "tokenizer.json",
    "tokenizer/tokenizer.json",
};

const zimage_tokenizer_config_files = [_][]const u8{
    "tokenizer_config.json",
    "tokenizer/tokenizer_config.json",
};

pub fn repositoryFiles(model_type: ModelType) RepositoryFiles {
    return switch (model_type) {
        .zimage => .{
            .model_index = "model_index.json",
            .tensor_entrypoints = &zimage_tensor_entrypoints,
            .text_encoder_config = "text_encoder/config.json",
            .transformer_config = "transformer/config.json",
            .vae_config = "vae/config.json",
            .scheduler_config = "scheduler/scheduler_config.json",
            .tokenizer = &zimage_tokenizer_files,
            .tokenizer_config = &zimage_tokenizer_config_files,
        },
    };
}

pub const Shardings = struct {
    model: zml.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        return .{
            .model = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth })),
        };
    }

    pub fn all(self: Shardings) [1]zml.Sharding {
        return .{self.model};
    }
};

pub fn parse_config(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}

pub fn parse_config_at_path(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8) !std.json.Parsed(T) {
    const file = try dir.openFile(io, path, .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}
