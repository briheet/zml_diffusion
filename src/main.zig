const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const models = @import("models.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
    .log_scope_levels = &.{
        .{ .scope = .@"zml/io", .level = .err },
        .{ .scope = .@"zml/module", .level = .err },
    },
};

const log = std.log.scoped(.diffusion);

const Args = struct {
    model: []const u8,
    prompt: ?[]const u8 = null,
    negative_prompt: []const u8 = "",
    guidance_scale: f32 = 5.0,
    cfg_normalization: bool = false,
    height: u32 = 1024,
    width: u32 = 1024,
    seqlen: u32 = 512,
    steps: u32 = 50,
    output: []const u8 = "zimage.png",

    pub const help =
        \\ Use diffusion --model_path [options]
        \\
        \\ Runs image generation with a model selected from `model_type` in the `config.toml`.
        \\
        \\ Options:
        \\   --model=<path>              Path to the model repository (required)
        \\   --prompt=<string>           Runs a single prompt for image generation (required)
        \\   --negative-prompt=<string>  Negative prompt used for classifier-free guidance (optional)
        \\   --guidance_scale=<float>    Classifier free guidance scale (optional)
        \\   --cfg-normalization         Caps the guided prediction norm to the positive prediction norm (optional)
        \\   --height=<pixel>            Output dimension. Should be divisible by 16. Defaults to 1024 (optional)
        \\   --width=<pixel>             Output dimension. Should be divisible by 16. Defaults to 1024 (optional)
        \\   --steps=<count>             Number of denoising steps. Defaults to 50. (optional)
        \\   --seqlen=<number>           Maximum tokenization prompt len. Defaults to 512. (optional)
        \\   --output=<path>             Image output path. Defaults to `zimage.png`. (optional)
    ;
};

fn loadTensorRegistry(
    allocator: std.mem.Allocator,
    io: std.Io,
    repo: std.Io.Dir,
    files: models.RepositoryFiles,
) !zml.safetensors.TensorRegistry {
    var registry = zml.safetensors.TensorRegistry.init(allocator);
    errdefer registry.deinit();

    for (files.tensor_entrypoints) |path| {
        const component_dir_path = std.fs.path.dirname(path) orelse ".";
        const entrypoint_name = std.fs.path.basename(path);
        const component_name = std.fs.path.dirname(path) orelse unreachable;

        var component_repo = try repo.openDir(io, component_dir_path, .{});
        defer component_repo.close(io);

        const entrypoint = try component_repo.openFile(io, entrypoint_name, .{ .mode = .read_only });
        defer entrypoint.close(io);

        var component_registry = try zml.safetensors.fetchRegistry(
            allocator,
            io,
            component_repo,
            entrypoint,
        );
        defer component_registry.deinit();

        try registry.mergeMetadata(component_registry.metadata);

        var it = component_registry.iterator();
        while (it.next()) |entry| {
            const prefixed_name = try std.fmt.allocPrint(allocator, "{s}.{s}", .{
                component_name,
                entry.value_ptr.name,
            });
            defer allocator.free(prefixed_name);

            var tensor = entry.value_ptr.*;
            tensor.name = prefixed_name;
            try registry.registerTensor(tensor);
        }
    }

    return registry;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    // `bazel run` executes binaries from Bazel's runfiles tree by default.
    // If available, switch back to the shell's original working directory.
    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = stdx.flags.parse(init.minimal.args, Args);

    //
    // Virtual File system
    //
    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var http_client: std.http.Client = .{ .allocator = allocator, .io = init.io };
    defer http_client.deinit();

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer hf_vfs.deinit();

    var s3_vfs: zml.io.VFS.S3 = try .auto(allocator, init.io, &http_client, init.environ_map);
    defer s3_vfs.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());
    try vfs.register("hf", hf_vfs.io());
    try vfs.register("s3", s3_vfs.io());

    const io = vfs.io();

    //
    // Platform and backend selection
    //
    const platform: *zml.Platform = try .auto(allocator, io, .{
        .cpu = .{ .device_count = 1 },
    });
    defer platform.deinit(allocator, io);

    log.info("\n{f}", .{platform.fmtVerbose()});

    //
    // Model initalization
    //
    log.info("Resolving model repository..", .{});
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    log.info("Initializing model..", .{});
    const model_type = try models.detectModelType(allocator, io, repo);
    const repository_files = models.repositoryFiles(model_type);

    var registry = try loadTensorRegistry(allocator, io, repo, repository_files);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    // Load model
    const model = try allocator.create(models.LoadedModel);
    model.* = models.LoadedModel.load(allocator, io, repo, store.view(), model_type) catch |err| {
        allocator.destroy(model);
        return err;
    };
    defer {
        model.deinit(allocator);
        allocator.destroy(model);
    }

    // Defines how the model's tensors are sharded across the available devices.
    const shardings: models.Shardings = try .init(platform);

    //
    // Load the model and compile it
    //
    var progress = std.Progress.start(io, .{ .root_name = args.model });
    errdefer progress.end();

    var inference_pipeline = switch (model.*) {
        .zimage => |*zimage_loaded_model| try models.pipeline.init(
            allocator,
            io,
            repo,
            platform,
            zimage_loaded_model,
            &store,
            shardings,
            args.seqlen,
            args.height,
            args.width,
            &progress,
        ),
    };
    defer inference_pipeline.deinit(allocator);

    progress.end();

    try printZmlLogo(io);

    const generation_started: std.Io.Timestamp = .now(io, .awake);
    try inference_pipeline.run(.{
        .prompt = args.prompt orelse return models.InferenceErrors.MissingPrompt,
        .negative_prompt = args.negative_prompt,
        .num_inference_steps = args.steps,
        .guidance_scale = args.guidance_scale,
        .cfg_normalization = args.cfg_normalization,
        .height = args.height,
        .width = args.width,
        .output_path = args.output,
    });
    log.info("Generated image [{f}]", .{generation_started.untilNow(io, .awake)});
}

pub fn printZmlLogo(io: std.Io) !void {
    const LOGO =
        \\
        \\
        \\ ███████╗███╗   ███╗██╗
        \\ ╚══███╔╝████╗ ████║██║
        \\   ███╔╝ ██╔████╔██║██║
        \\  ███╔╝  ██║╚██╔╝██║██║  .ai
        \\ ███████╗██║ ╚═╝ ██║███████╗
        \\ ╚══════╝╚═╝     ╚═╝╚══════╝
        \\
        \\
        \\
    ;
    var writer = std.Io.File.stdout().writerStreaming(io, &.{});
    try writer.interface.writeAll(LOGO);
    try writer.interface.flush();
}
