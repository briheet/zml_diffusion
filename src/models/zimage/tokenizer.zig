const std = @import("std");

const zml = @import("zml");
const common = @import("../common.zig");

pub const AddedToken = struct {
    content: []const u8,
    lstrip: bool = false,
    normalized: bool = false,
    rstrip: bool = false,
    single_word: bool = false,
    special: bool = false,
};

pub const AddedTokenDecoderEntry = struct {
    id: u32,
    token: AddedToken,
};

pub const Config = struct {
    add_bos_token: bool = false,
    add_prefix_space: bool = false,
    added_tokens_decoder: []const AddedTokenDecoderEntry = &.{},
    additional_special_tokens: []const []const u8 = &.{},
    bos_token: ?[]const u8 = null,
    chat_template: []const u8 = "",
    clean_up_tokenization_spaces: bool = false,
    eos_token: []const u8 = "<|im_end|>",
    errors: []const u8 = "replace",
    model_max_length: usize = 131072,
    pad_token: []const u8 = "<|endoftext|>",
    split_special_tokens: bool = false,
    tokenizer_class: []const u8 = "Qwen2Tokenizer",
    unk_token: ?[]const u8 = null,
};

const ConfigOverrides = struct {
    additional_special_tokens: []const []const u8 = &.{},
    bos_token: ?[]const u8 = null,
    chat_template: []const u8 = "",
    eos_token: ?[]const u8 = null,
    pad_token: ?[]const u8 = null,
    tokenizer_class: ?[]const u8 = null,
    unk_token: ?[]const u8 = null,
};

pub const PromptEncoding = struct {
    ids: zml.Slice,
    mask: zml.Slice,

    pub fn deinit(self: *PromptEncoding, allocator: std.mem.Allocator) void {
        self.ids.free(allocator);
        self.mask.free(allocator);
    }
};

pub const Tokenizer = struct {
    inner: zml.tokenizer.Tokenizer,
    config: Config,
    parsed_config: ?std.json.Parsed(ConfigOverrides) = null,

    pub fn init(inner: zml.tokenizer.Tokenizer, config: Config) Tokenizer {
        return .{
            .inner = inner,
            .config = config,
            .parsed_config = null,
        };
    }

    pub fn fromBytes(allocator: std.mem.Allocator, bytes: []const u8, config: Config) !Tokenizer {
        return .{
            .inner = try zml.tokenizer.Tokenizer.fromBytes(allocator, bytes),
            .config = config,
            .parsed_config = null,
        };
    }

    pub fn fromDir(
        allocator: std.mem.Allocator,
        io: std.Io,
        dir: std.Io.Dir,
        files: common.RepositoryFiles,
        config: Config,
    ) !Tokenizer {
        const bytes = b: {
            for (files.tokenizer) |path| {
                const file = dir.openFile(io, path, .{}) catch |err| switch (err) {
                    error.FileNotFound => continue,
                    else => return err,
                };
                defer file.close(io);

                var reader = file.reader(io, &.{});
                break :b try reader.interface.readAlloc(allocator, try file.length(io));
            }

            return error.FileNotFound;
        };
        defer allocator.free(bytes);

        const parsed_config = cfg: {
            for (files.tokenizer_config) |path| {
                const file = dir.openFile(io, path, .{}) catch |err| switch (err) {
                    error.FileNotFound => continue,
                    else => return err,
                };
                defer file.close(io);

                var reader = file.reader(io, &.{});
                const config_bytes = try reader.interface.readAlloc(allocator, try file.length(io));
                defer allocator.free(config_bytes);

                break :cfg try std.json.parseFromSlice(ConfigOverrides, allocator, config_bytes, .{
                    .ignore_unknown_fields = true,
                });
            }

            break :cfg null;
        };

        const merged_config = if (parsed_config) |parsed| blk: {
            var merged = config;
            const overrides = parsed.value;
            merged.additional_special_tokens = overrides.additional_special_tokens;
            merged.bos_token = overrides.bos_token;
            if (overrides.chat_template.len > 0) merged.chat_template = overrides.chat_template;
            if (overrides.eos_token) |value| merged.eos_token = value;
            if (overrides.pad_token) |value| merged.pad_token = value;
            if (overrides.tokenizer_class) |value| merged.tokenizer_class = value;
            merged.unk_token = overrides.unk_token;
            break :blk merged;
        } else config;

        return .{
            .inner = try zml.tokenizer.Tokenizer.fromBytes(allocator, bytes),
            .config = merged_config,
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *Tokenizer) void {
        if (self.parsed_config) |*parsed| parsed.deinit();
        self.inner.deinit();
    }

    pub fn encoder(self: *const Tokenizer) !zml.tokenizer.Tokenizer.Encoder {
        return self.inner.encoder();
    }

    pub fn decoder(self: *const Tokenizer) !zml.tokenizer.Tokenizer.Decoder {
        return self.inner.decoder();
    }

    pub fn tokenId(self: *const Tokenizer, token: []const u8) ?u32 {
        return self.inner.tokenId(token);
    }

    pub fn bosTokenId(self: *const Tokenizer) ?u32 {
        return if (self.config.bos_token) |token| self.tokenId(token) else null;
    }

    pub fn eosTokenId(self: *const Tokenizer) ?u32 {
        return self.tokenId(self.config.eos_token);
    }

    pub fn padTokenId(self: *const Tokenizer) ?u32 {
        return self.tokenId(self.config.pad_token);
    }

    pub fn unkTokenId(self: *const Tokenizer) ?u32 {
        return if (self.config.unk_token) |token| self.tokenId(token) else null;
    }

    pub fn encodePromptAlloc(
        self: *const Tokenizer,
        allocator: std.mem.Allocator,
        prompt: []const u8,
        max_sequence_length: u32,
    ) !PromptEncoding {
        const rendered_prompt = try self.applyUserChatTemplateAlloc(allocator, prompt);
        defer allocator.free(rendered_prompt);

        var prompt_encoder = try self.encoder();
        defer prompt_encoder.deinit();

        const prompt_tokens = try prompt_encoder.encodeAlloc(allocator, rendered_prompt);
        defer allocator.free(prompt_tokens);

        if (prompt_tokens.len > max_sequence_length) return error.PromptTooLong;

        const token_shape = zml.Shape.init(.{ .b = 1, .s = max_sequence_length }, .u32);
        var ids = try zml.Slice.alloc(allocator, token_shape);
        errdefer ids.free(allocator);
        const token_items = ids.items(u32);
        @memset(token_items, self.padTokenId() orelse 0);
        @memcpy(token_items[0..prompt_tokens.len], prompt_tokens);

        const mask_shape = zml.Shape.init(.{ .b = 1, .s = max_sequence_length }, .bool);
        var mask = try zml.Slice.alloc(allocator, mask_shape);
        errdefer mask.free(allocator);
        const mask_items = mask.items(bool);
        @memset(mask_items, false);
        @memset(mask_items[0..prompt_tokens.len], true);

        return .{
            .ids = ids,
            .mask = mask,
        };
    }

    pub fn applyUserChatTemplateAlloc(self: *const Tokenizer, allocator: std.mem.Allocator, prompt: []const u8) ![]u8 {
        _ = self;
        return std.fmt.allocPrint(allocator, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n", .{prompt});
    }
};
