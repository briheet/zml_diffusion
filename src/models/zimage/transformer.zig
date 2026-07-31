const std = @import("std");

const zml = @import("zml");

pub const ADALN_EMBED_DIM: u32 = 256;
pub const SEQ_MULTI_OF: u32 = 32;
pub const X_PAD_DIM: u32 = 64;
const NumRefinerLayers = 2;
const NumLayers = 30;

// It uses ZImageTransformer2DModel.
pub const Transformer = struct {
    pub const Config = struct {
        all_f_patch_size: [1]u32 = .{1},
        all_patch_size: [1]u32 = .{2},
        axes_dims: [3]u32 = .{ 32, 48, 48 },
        axes_lens: [3]u32 = .{ 1536, 512, 512 },

        cap_feat_dim: u32 = 2560,
        dim: u32 = 3840,
        in_channels: u32 = 16,
        n_heads: u32 = 30,
        n_kv_heads: u32 = 30,
        n_layers: u32 = 30,
        n_refiner_layers: u32 = 2,
        norm_eps: f32 = 1e-5,
        qk_norm: bool = true,
        rope_theta: f32 = 256.0,
        siglip_feat_dim: ?u32 = null,
        t_scale: f32 = 1000.0,
    };

    in_channels: u32,
    out_channels: u32,
    all_patch_size: [1]u32,
    all_f_patch_size: [1]u32,
    dim: u32,
    n_heads: u32,
    rope_theta: f32,
    t_scale: f32,
    gradient_checkpointing: bool,

    all_x_embedder: []zml.nn.Linear,
    all_final_layer: []FinalLayer,
    noise_refiner: []ZImageTransformerBlock,
    context_refiner: []ZImageTransformerBlock,
    t_embedder: TimestepEmbedder,
    cap_embedder: Embedder,

    siglip_embedder: ?Embedder,
    siglip_refiner: ?[]ZImageTransformerBlock,
    siglip_pad_token: ?zml.Tensor,

    x_pad_token: zml.Tensor,
    cap_pad_token: zml.Tensor,

    layers: []ZImageTransformerBlock,
    axes_dims: [3]u32,
    axes_lens: [3]u32,
    rope_embedder: RopeEmbedder,

    pub fn init(
        self: *Transformer,
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Config,
    ) !void {
        std.debug.assert(config.all_patch_size.len == config.all_f_patch_size.len);
        std.debug.assert(@divExact(config.dim, config.n_heads) == config.axes_dims[0] + config.axes_dims[1] + config.axes_dims[2]);

        self.in_channels = config.in_channels;
        self.out_channels = config.in_channels;
        self.all_patch_size = config.all_patch_size;
        self.all_f_patch_size = config.all_f_patch_size;
        self.dim = config.dim;
        self.n_heads = config.n_heads;
        self.rope_theta = config.rope_theta;
        self.t_scale = config.t_scale;
        self.gradient_checkpointing = false;
        self.axes_dims = config.axes_dims;
        self.axes_lens = config.axes_lens;
        self.rope_embedder = RopeEmbedder.init(config.rope_theta, config.axes_dims, config.axes_lens);

        self.all_x_embedder = try allocator.alloc(zml.nn.Linear, config.all_patch_size.len);
        errdefer allocator.free(self.all_x_embedder);
        self.all_final_layer = try allocator.alloc(FinalLayer, config.all_patch_size.len);
        errdefer allocator.free(self.all_final_layer);

        for (config.all_patch_size, config.all_f_patch_size, 0..) |patch_size, f_patch_size, i| {
            var key_buffer: [32]u8 = undefined;
            const key = try std.fmt.bufPrint(&key_buffer, "{d}-{d}", .{ patch_size, f_patch_size });
            const inner_dim = f_patch_size * patch_size * patch_size * config.in_channels;
            const final_dim = patch_size * patch_size * f_patch_size * config.in_channels;

            self.all_x_embedder[i] = .init(
                store.withPrefix("all_x_embedder").withPrefix(key).createTensor("weight", .{ .dout, .d }, .{
                    .dout = .model,
                    .d = .replicated,
                }),
                store.withPrefix("all_x_embedder").withPrefix(key).createTensor("bias", .{.dout}, .{
                    .dout = .model,
                }),
                .d,
            );

            self.all_final_layer[i] = try FinalLayer.init(
                store.withPrefix("all_final_layer").withPrefix(key),
                final_dim,
            );

            _ = inner_dim;
        }

        std.debug.assert(config.n_refiner_layers == NumRefinerLayers);
        std.debug.assert(config.n_layers == NumLayers);

        self.noise_refiner = try allocator.alloc(ZImageTransformerBlock, config.n_refiner_layers);
        errdefer allocator.free(self.noise_refiner);
        for (self.noise_refiner, 0..) |*layer, i| {
            layer.* = try ZImageTransformerBlock.init(
                store.withPrefix("noise_refiner").withLayer(i),
                1000 + @as(u32, @intCast(i)),
                config.dim,
                config.n_heads,
                config.n_kv_heads,
                config.norm_eps,
                config.qk_norm,
                true,
            );
        }

        self.context_refiner = try allocator.alloc(ZImageTransformerBlock, config.n_refiner_layers);
        errdefer allocator.free(self.context_refiner);
        for (self.context_refiner, 0..) |*layer, i| {
            layer.* = try ZImageTransformerBlock.init(
                store.withPrefix("context_refiner").withLayer(i),
                @intCast(i),
                config.dim,
                config.n_heads,
                config.n_kv_heads,
                config.norm_eps,
                config.qk_norm,
                false,
            );
        }

        self.layers = try allocator.alloc(ZImageTransformerBlock, config.n_layers);
        errdefer allocator.free(self.layers);
        for (self.layers, 0..) |*layer, i| {
            layer.* = try ZImageTransformerBlock.init(
                store.withPrefix("layers").withLayer(i),
                @intCast(i),
                config.dim,
                config.n_heads,
                config.n_kv_heads,
                config.norm_eps,
                config.qk_norm,
                true,
            );
        }

        self.siglip_embedder = null;
        self.siglip_refiner = null;
        self.siglip_pad_token = null;
        if (config.siglip_feat_dim) |siglip_feat_dim| {
            self.siglip_embedder = try Embedder.init(
                store.withPrefix("siglip_embedder"),
                config.norm_eps,
            );
            _ = siglip_feat_dim;

            const refiner = try allocator.alloc(ZImageTransformerBlock, config.n_refiner_layers);
            errdefer allocator.free(refiner);
            for (refiner, 0..) |*layer, i| {
                layer.* = try ZImageTransformerBlock.init(
                    store.withPrefix("siglip_refiner").withLayer(i),
                    2000 + @as(u32, @intCast(i)),
                    config.dim,
                    config.n_heads,
                    config.n_kv_heads,
                    config.norm_eps,
                    config.qk_norm,
                    false,
                );
            }
            self.siglip_refiner = refiner;
            self.siglip_pad_token = store.createTensor("siglip_pad_token", .{ .tok, .d }, .{
                .tok = .replicated,
                .d = .replicated,
            });
        }

        self.t_embedder = try TimestepEmbedder.init(store.withPrefix("t_embedder"), @min(config.dim, ADALN_EMBED_DIM), 1024);
        self.cap_embedder = try Embedder.init(store.withPrefix("cap_embedder"), config.norm_eps);
        self.x_pad_token = store.createTensor("x_pad_token", .{ .tok, .d }, .{
            .tok = .replicated,
            .d = .replicated,
        });
        self.cap_pad_token = store.createTensor("cap_pad_token", .{ .tok, .d }, .{
            .tok = .replicated,
            .d = .replicated,
        });
    }

    pub fn deinit(self: *Transformer, allocator: std.mem.Allocator) void {
        allocator.free(self.all_x_embedder);
        allocator.free(self.all_final_layer);
        allocator.free(self.noise_refiner);
        allocator.free(self.context_refiner);
        if (self.siglip_refiner) |refiner| allocator.free(refiner);
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Transformer), allocator: std.mem.Allocator) void {
        for (self.all_x_embedder) |*embedder| {
            embedder.weight.deinit();
            if (embedder.bias) |*bias| bias.deinit();
        }
        for (self.all_final_layer) |*layer| FinalLayer.unloadBuffers(layer);
        for (self.noise_refiner) |*layer| ZImageTransformerBlock.unloadBuffers(layer);
        for (self.context_refiner) |*layer| ZImageTransformerBlock.unloadBuffers(layer);
        TimestepEmbedder.unloadBuffers(&self.t_embedder);
        Embedder.unloadBuffers(&self.cap_embedder);
        if (self.siglip_embedder) |*embedder| Embedder.unloadBuffers(embedder);
        if (self.siglip_refiner) |layers| {
            for (layers) |*layer| ZImageTransformerBlock.unloadBuffers(layer);
        }
        if (self.siglip_pad_token) |*token| token.deinit();
        self.x_pad_token.deinit();
        self.cap_pad_token.deinit();
        for (self.layers) |*layer| ZImageTransformerBlock.unloadBuffers(layer);

        allocator.free(self.all_x_embedder);
        allocator.free(self.all_final_layer);
        allocator.free(self.noise_refiner);
        allocator.free(self.context_refiner);
        if (self.siglip_refiner) |refiner| allocator.free(refiner);
        allocator.free(self.layers);
    }

    pub fn forward(
        self: *const Transformer,
        image: zml.Tensor,
        t: zml.Tensor,
        cap_feat: zml.Tensor,
        patch_size: u32,
        f_patch_size: u32,
    ) zml.Tensor {
        std.debug.assert(patch_size == self.all_patch_size[0]);
        std.debug.assert(f_patch_size == self.all_f_patch_size[0]);
        const size = ImageSize{
            .f = @intCast(image.dim(.f)),
            .h = @intCast(image.dim(.h)),
            .w = @intCast(image.dim(.w)),
        };

        const timestep_embed = self.t_embedder.forward(t.convert(.f32).scale(self.t_scale)).squeeze(.b);
        const x_patches = patchifyImage(image, patch_size, f_patch_size).convert(self.all_x_embedder[0].weight.dtype());
        const cap_len: u32 = @intCast(cap_feat.dim(.s));
        const padded_cap_len = std.mem.alignForward(u32, cap_len, SEQ_MULTI_OF);
        const x_token_count: u32 = @intCast(x_patches.dim(.s));
        const padded_x_token_count = std.mem.alignForward(u32, x_token_count, SEQ_MULTI_OF);

        var x_pos_ids = createCoordinateGrid(
            .{
                @divExact(size.f, f_patch_size),
                @divExact(size.h, patch_size),
                @divExact(size.w, patch_size),
            },
            .{ padded_cap_len + 1, 0, 0 },
        );
        var x_hidden = self.all_x_embedder[0].forward(x_patches).rename(.{ .dout = .d });
        if (padded_x_token_count > x_token_count) {
            const pad_len = padded_x_token_count - x_token_count;
            const pad_shape = zml.Shape.init(.{ .s = pad_len, .d = self.dim }, self.x_pad_token.dtype());
            const pad_tokens = self.x_pad_token.rename(.{ .tok = .s }).broad(pad_shape);
            x_hidden = zml.Tensor.concatenate(&.{ x_hidden, pad_tokens }, .s);
            x_pos_ids = zml.Tensor.concatenate(&.{
                x_pos_ids,
                zml.Tensor.zeroes(.init(.{ .s = pad_len, .coord = 3 }, .i32)),
            }, .s);
        }
        const x_freqs = self.rope_embedder.forward(x_pos_ids);

        for (self.noise_refiner) |layer| {
            x_hidden = layer.forward(x_hidden, null, x_freqs, timestep_embed, null, null, null);
        }

        var cap_hidden = self.cap_embedder.forward(cap_feat).rename(.{ .dout = .d });
        if (padded_cap_len > cap_len) {
            const pad_shape = zml.Shape.init(.{ .s = padded_cap_len - cap_len, .d = self.dim }, self.cap_pad_token.dtype());
            const pad_tokens = self.cap_pad_token.rename(.{ .tok = .s }).broad(pad_shape);
            cap_hidden = zml.Tensor.concatenate(&.{ cap_hidden, pad_tokens }, .s);
        }
        const cap_pos_ids = createCoordinateGrid(.{ padded_cap_len, 1, 1 }, .{ 1, 0, 0 });
        const cap_freqs = self.rope_embedder.forward(cap_pos_ids);

        for (self.context_refiner) |layer| {
            cap_hidden = layer.forward(cap_hidden, null, cap_freqs, null, null, null, null);
        }

        var unified = zml.Tensor.concatenate(&.{ x_hidden, cap_hidden }, .s);
        const unified_freqs = zml.Tensor.concatenate(&.{ x_freqs, cap_freqs }, .s);

        for (self.layers) |layer| {
            unified = layer.forward(unified, null, unified_freqs, timestep_embed, null, null, null);
        }

        unified = self.all_final_layer[0].forward(unified, timestep_embed, null, null, null);
        const image_tokens = unified.slice1d(.s, .{ .end = x_patches.dim(.s) });
        return unpatchifyOne(image_tokens, size, patch_size, f_patch_size, self.out_channels);
    }

    pub fn denoiseStep(self: *const Transformer, latent: zml.Tensor, timestep: zml.Tensor, prompt_embeds: zml.Tensor) zml.Tensor {
        return self.forward(
            latent,
            timestep,
            prompt_embeds,
            self.all_patch_size[0],
            self.all_f_patch_size[0],
        );
    }
};

pub const TimestepEmbedder = struct {
    mlp_1: zml.nn.Linear,
    mlp_2: zml.nn.Linear,
    frequency_embedding_size: u32,

    pub fn init(store: zml.io.TensorStore.View, out_size: u32, mid_size: u32) !TimestepEmbedder {
        return .{
            .mlp_1 = .init(
                store.withPrefix("mlp").withLayer(0).createTensor("weight", .{ .dout, .d }, .{
                    .dout = .model,
                    .d = .replicated,
                }),
                store.withPrefix("mlp").withLayer(0).createTensor("bias", .{.dout}, .{
                    .dout = .model,
                }),
                .d,
            ),
            .mlp_2 = .init(
                store.withPrefix("mlp").withLayer(2).createTensor("weight", .{ .dout, .d }, .{
                    .dout = .model,
                    .d = .replicated,
                }),
                store.withPrefix("mlp").withLayer(2).createTensor("bias", .{.dout}, .{
                    .dout = .model,
                }),
                .d,
            ),
            .frequency_embedding_size = @max(out_size, mid_size) - (@max(out_size, mid_size) - 256),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TimestepEmbedder)) void {
        self.mlp_1.weight.deinit();
        if (self.mlp_1.bias) |*b| b.deinit();
        self.mlp_2.weight.deinit();
        if (self.mlp_2.bias) |*b| b.deinit();
    }

    pub fn timestepEmbedding(t: zml.Tensor, dim: u32, max_period: f32) zml.Tensor {
        const half = @divFloor(dim, 2);
        const freqs = zml.Tensor.arange(.{ .end = half }, .f32)
            .withTags(.{.f})
            .scale(-std.math.log(f32, std.math.e, max_period) / @as(f32, @floatFromInt(half)))
            .exp();
        const shape = zml.Shape.init(.{ .b = t.dim(.b), .f = half }, .f32);
        const t_expanded = t.convert(.f32).insertAxes(.b, .{.f}).broad(shape);
        const freqs_expanded = freqs.insertAxes(.f, .{.b}).broad(shape);
        const args = t_expanded.mul(freqs_expanded);
        var embedding = zml.Tensor.concatenate(&.{ args.cos(), args.sin() }, -1);
        if (@mod(dim, 2) != 0) {
            const zero = zml.Tensor.constant(.{ .f32 = 0 }).broad(zml.Shape.init(.{ .b = t.dim(.b), .pad = 1 }, .f32));
            embedding = zml.Tensor.concatenate(&.{ embedding, zero }, -1);
        }
        return embedding;
    }

    pub fn forward(self: TimestepEmbedder, t: zml.Tensor) zml.Tensor {
        const t_freq = timestepEmbedding(t, self.frequency_embedding_size, 10000)
            .rename(.{ .f = .d })
            .convert(self.mlp_1.weight.dtype());
        const hidden = self.mlp_1.forward(t_freq).rename(.{ .dout = .d }).silu();
        return self.mlp_2.forward(hidden).rename(.{ .dout = .d });
    }
};

pub const Embedder = struct {
    norm: RMSNorm,
    linear: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View, norm_eps: f32) !Embedder {
        return .{
            .norm = .init(store.withPrefix("0"), norm_eps, true, false),
            .linear = .init(
                store.withPrefix("1").createTensor("weight", .{ .dout, .d }, .{
                    .dout = .model,
                    .d = .replicated,
                }),
                store.withPrefix("1").createTensor("bias", .{.dout}, .{
                    .dout = .model,
                }),
                .d,
            ),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Embedder)) void {
        RMSNorm.unloadBuffers(&self.norm);
        self.linear.weight.deinit();
        if (self.linear.bias) |*b| b.deinit();
    }

    pub fn forward(self: Embedder, x: zml.Tensor) zml.Tensor {
        return self.linear.forward(self.norm.forward(x).convert(self.linear.weight.dtype()));
    }
};

pub const RMSNorm = struct {
    weight: ?zml.Tensor = null,
    bias: ?zml.Tensor = null,
    eps: f32,

    pub fn init(
        store: zml.io.TensorStore.View,
        eps: f32,
        elementwise_affine: bool,
        with_bias: bool,
    ) RMSNorm {
        return .{
            .weight = if (elementwise_affine)
                store.createTensor("weight", .{.d}, .{ .d = .replicated })
            else
                null,
            .bias = if (elementwise_affine and with_bias)
                store.createTensor("bias", .{.d}, .{ .d = .replicated })
            else
                null,
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RMSNorm)) void {
        if (self.weight) |*w| w.deinit();
        if (self.bias) |*b| b.deinit();
    }

    pub fn forward(self: RMSNorm, hidden_states: zml.Tensor) zml.Tensor {
        const input_dtype = hidden_states.dtype();
        const hidden_f32 = hidden_states.convert(.f32);
        var out = zml.nn.rmsNorm(hidden_f32, .d, self.eps);

        if (self.weight) |weight| {
            out = out.mul(weight.convert(.f32).broadcast(out.shape(), &.{out.axis(-1)}));
        } else {
            out = out.convert(input_dtype);
        }

        if (self.bias) |bias| {
            out = out.add(bias.convert(.f32).broadcast(out.shape(), &.{out.axis(-1)}));
        }

        return out.convert(input_dtype);
    }
};

pub const FinalLayer = struct {
    norm_final: OptionalLayerNorm,
    linear: zml.nn.Linear,
    adaLN_modulation: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View, out_channels: u32) !FinalLayer {
        _ = out_channels;
        return .{
            .norm_final = .{
                .weight = store.withPrefix("norm_final").maybeCreateTensor("weight", .{.d}, .replicated),
                .bias = store.withPrefix("norm_final").maybeCreateTensor("bias", .{.d}, .replicated),
                .eps = 1e-6,
            },
            .linear = .init(
                store.withPrefix("linear").createTensor("weight", .{ .dout, .d }, .{
                    .dout = .model,
                    .d = .replicated,
                }),
                store.withPrefix("linear").createTensor("bias", .{.dout}, .{ .dout = .model }),
                .d,
            ),
            .adaLN_modulation = .init(
                store.withPrefix("adaLN_modulation").withLayer(1).createTensor("weight", .{ .dout, .d }, .{
                    .dout = .model,
                    .d = .replicated,
                }),
                store.withPrefix("adaLN_modulation").withLayer(1).createTensor("bias", .{.dout}, .{
                    .dout = .model,
                }),
                .d,
            ),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(FinalLayer)) void {
        OptionalLayerNorm.unloadBuffers(&self.norm_final);
        self.linear.weight.deinit();
        if (self.linear.bias) |*b| b.deinit();
        self.adaLN_modulation.weight.deinit();
        if (self.adaLN_modulation.bias) |*b| b.deinit();
    }

    pub fn forward(
        self: FinalLayer,
        x: zml.Tensor,
        c: ?zml.Tensor,
        noise_mask: ?zml.Tensor,
        c_noisy: ?zml.Tensor,
        c_clean: ?zml.Tensor,
    ) zml.Tensor {
        const seq_len = x.dim(.s);

        const scale = if (noise_mask) |mask| blk: {
            const scale_noisy = self.adaLN_modulation.forward((c_noisy orelse @panic("missing c_noisy")).silu()).rename(.{ .dout = .d }).addConstant(1.0);
            const scale_clean = self.adaLN_modulation.forward((c_clean orelse @panic("missing c_clean")).silu()).rename(.{ .dout = .d }).addConstant(1.0);
            break :blk selectPerToken(scale_noisy, scale_clean, mask, @intCast(seq_len));
        } else blk: {
            const global = self.adaLN_modulation.forward((c orelse @panic("missing c")).silu()).rename(.{ .dout = .d }).addConstant(1.0);
            break :blk global.insertAxes(.d, .{.s});
        };

        const normed = self.norm_final.forward(x).mul(scale);
        return self.linear.forward(normed).rename(.{ .dout = .d });
    }
};

const OptionalLayerNorm = struct {
    weight: ?zml.Tensor,
    bias: ?zml.Tensor = null,
    eps: f32 = 1e-5,

    pub fn unloadBuffers(self: *zml.Bufferized(OptionalLayerNorm)) void {
        if (self.weight) |*weight| weight.deinit();
        if (self.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: OptionalLayerNorm, x: zml.Tensor) zml.Tensor {
        const normed = zml.nn.normalizeVariance(x, self.eps);
        const ax = x.axis(-1);
        var out = normed;
        if (self.weight) |weight| out = out.mul(weight.broadcast(x.shape(), &.{ax}));
        if (self.bias) |bias| out = out.add(bias.broadcast(x.shape(), &.{ax}));
        return out;
    }
};

fn selectPerToken(
    value_noisy: zml.Tensor,
    value_clean: zml.Tensor,
    noise_mask: zml.Tensor,
    seq_len: u32,
) zml.Tensor {
    if (value_noisy.shape().hasTags(.{.b})) {
        const noisy = value_noisy.insertAxes(.d, .{.s}).broad(zml.Shape.init(.{ .b = value_noisy.dim(.b), .s = seq_len, .d = value_noisy.dim(.d) }, value_noisy.dtype()));
        const clean = value_clean.insertAxes(.d, .{.s}).broad(zml.Shape.init(.{ .b = value_clean.dim(.b), .s = seq_len, .d = value_clean.dim(.d) }, value_clean.dtype()));
        const mask = noise_mask.insertAxes(.s, .{.d}).broad(zml.Shape.init(.{ .b = noise_mask.dim(.b), .s = seq_len, .d = value_noisy.dim(.d) }, noise_mask.dtype()));
        return mask.cmp(.EQ, zml.Tensor.scalar(1, mask.dtype())).select(noisy, clean);
    }

    const noisy = value_noisy.insertAxes(.d, .{.s}).broad(zml.Shape.init(.{ .s = seq_len, .d = value_noisy.dim(.d) }, value_noisy.dtype()));
    const clean = value_clean.insertAxes(.d, .{.s}).broad(zml.Shape.init(.{ .s = seq_len, .d = value_clean.dim(.d) }, value_clean.dtype()));
    const mask = noise_mask.insertAxes(.s, .{.d}).broad(zml.Shape.init(.{ .s = seq_len, .d = value_noisy.dim(.d) }, noise_mask.dtype()));
    return mask.cmp(.EQ, zml.Tensor.scalar(1, mask.dtype())).select(noisy, clean);
}

pub const RopeEmbedder = struct {
    theta: f32,
    axes_dims: [3]u32,
    axes_lens: [3]u32,

    pub fn init(theta: f32, axes_dims: [3]u32, axes_lens: [3]u32) RopeEmbedder {
        return .{
            .theta = theta,
            .axes_dims = axes_dims,
            .axes_lens = axes_lens,
        };
    }

    pub fn precomputeFreqsCis(
        allocator: std.mem.Allocator,
        dim: []const u32,
        end: []const u32,
        theta: f32,
    ) ![]zml.Tensor {
        std.debug.assert(dim.len == end.len);
        const freqs_cis = try allocator.alloc(zml.Tensor, dim.len);
        errdefer allocator.free(freqs_cis);

        for (dim, end, 0..) |d, e, i| {
            const half = @divFloor(d, 2);
            const freqs = zml.Tensor.arange(.{ .end = half }, .f32)
                .withTags(.{.hd})
                .scale(-std.math.log(f32, std.math.e, theta) / @as(f32, @floatFromInt(d)))
                .exp();
            const timestep = zml.Tensor.arange(.{ .end = e }, .f32).withTags(.{.s});
            const args = timestep.outer(freqs);
            freqs_cis[i] = zml.Tensor.concatenate(&.{ args.cos(), args.sin() }, -1);
        }

        return freqs_cis;
    }

    pub fn forward(self: RopeEmbedder, ids: zml.Tensor) zml.Tensor {
        std.debug.assert(ids.rank() == 2);

        var result: [3]zml.Tensor = undefined;
        inline for (0..3) |i| {
            const axis_ids = ids.slice1d(-1, .single(i));
            const half = @divFloor(self.axes_dims[i], 2);
            const freqs = zml.Tensor.arange(.{ .end = half }, .f32)
                .withTags(.{.hd})
                .scale(-2.0 * std.math.log(f32, std.math.e, self.theta) / @as(f32, @floatFromInt(self.axes_dims[i])))
                .exp();
            result[i] = axis_ids.convert(.f32).outer(freqs);
        }

        return zml.Tensor.concatenate(&result, -1);
    }
};

const ImageSize = struct {
    f: u32,
    h: u32,
    w: u32,
};

fn createCoordinateGrid(size: [3]u32, start: [3]u32) zml.Tensor {
    const shape = zml.Shape.init(.{ .ft = size[0], .ht = size[1], .wt = size[2] }, .i32);
    const f = zml.Tensor.iota(shape, .ft).addConstant(@as(i32, @intCast(start[0])));
    const h = zml.Tensor.iota(shape, .ht).addConstant(@as(i32, @intCast(start[1])));
    const w = zml.Tensor.iota(shape, .wt).addConstant(@as(i32, @intCast(start[2])));
    return zml.Tensor.stack(&.{ f, h, w }, .last, .coord).merge(.{ .s = .{ .ft, .ht, .wt } });
}

fn patchifyImage(image: zml.Tensor, patch_size: u32, f_patch_size: u32) zml.Tensor {
    return image
        .splitAxis(.f, .{ .ft = .auto, .pf = f_patch_size })
        .splitAxis(.h, .{ .ht = .auto, .ph = patch_size })
        .splitAxis(.w, .{ .wt = .auto, .pw = patch_size })
        .transpose(.{ .ft, .ht, .wt, .pf, .ph, .pw, .c })
        .merge(.{ .s = .{ .ft, .ht, .wt }, .d = .{ .pf, .ph, .pw, .c } });
}

fn unpatchifyOne(
    x: zml.Tensor,
    size: ImageSize,
    patch_size: u32,
    f_patch_size: u32,
    out_channels: u32,
) zml.Tensor {
    const f_tokens = @divExact(size.f, f_patch_size);
    const h_tokens = @divExact(size.h, patch_size);
    const w_tokens = @divExact(size.w, patch_size);

    return x
        .splitAxis(.s, .{ .ft = f_tokens, .ht = h_tokens, .wt = w_tokens })
        .splitAxis(.d, .{ .pf = f_patch_size, .ph = patch_size, .pw = patch_size, .c = out_channels })
        .transpose(.{ .c, .ft, .pf, .ht, .ph, .wt, .pw })
        .merge(.{ .f = .{ .ft, .pf }, .h = .{ .ht, .ph }, .w = .{ .wt, .pw } });
}

pub const FeedForward = struct {
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,
    w3: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) FeedForward {
        return .{
            .w1 = .init(
                store.withPrefix("w1").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .w2 = .init(
                store.withPrefix("w2").createTensor("weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .model }),
                null,
                .dout,
            ),
            .w3 = .init(
                store.withPrefix("w3").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(FeedForward)) void {
        self.w1.weight.deinit();
        self.w2.weight.deinit();
        self.w3.weight.deinit();
    }

    pub fn forward(self: FeedForward, x: zml.Tensor) zml.Tensor {
        return self.w2.forward(self.w1.forward(x).silu().mul(self.w3.forward(x)));
    }
};

pub const ZSingleStreamAttention = struct {
    to_q: zml.nn.Linear,
    to_k: zml.nn.Linear,
    to_v: zml.nn.Linear,
    to_out: zml.nn.Linear,
    norm_q: ?RMSNorm,
    norm_k: ?RMSNorm,
    heads: u32,
    head_dim: u32,

    pub fn init(store: zml.io.TensorStore.View, dim: u32, n_heads: u32, qk_norm: bool) ZSingleStreamAttention {
        return .{
            .to_q = .init(
                store.withPrefix("to_q").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .to_k = .init(
                store.withPrefix("to_k").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .to_v = .init(
                store.withPrefix("to_v").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .to_out = .init(
                store.withPrefix("to_out").withLayer(0).createTensor("weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .model }),
                null,
                .dout,
            ),
            .norm_q = if (qk_norm) RMSNorm.init(store.withPrefix("norm_q"), 1e-5, true, false) else null,
            .norm_k = if (qk_norm) RMSNorm.init(store.withPrefix("norm_k"), 1e-5, true, false) else null,
            .heads = n_heads,
            .head_dim = @divExact(dim, n_heads),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ZSingleStreamAttention)) void {
        self.to_q.weight.deinit();
        self.to_k.weight.deinit();
        self.to_v.weight.deinit();
        self.to_out.weight.deinit();
        if (self.norm_q) |*norm| RMSNorm.unloadBuffers(norm);
        if (self.norm_k) |*norm| RMSNorm.unloadBuffers(norm);
    }

    fn applyFreqs(x: zml.Tensor, freqs_cis: zml.Tensor, seq_tag: zml.Shape.Tag) zml.Tensor {
        const paired = x.splitAxis(.hd, .{ .pair = .auto, .ri = 2 });
        const real = paired.slice1d(.ri, .single(0));
        const imag = paired.slice1d(.ri, .single(1));
        const phases = freqs_cis.rename(.{ .s = seq_tag, .hd = .pair }).convert(x.dtype());
        const cos = phases.cos().insertAxes(.pair, .{.h}).broad(real.shape());
        const sin = phases.sin().insertAxes(.pair, .{.h}).broad(real.shape());
        const rotated_real = real.mul(cos).sub(imag.mul(sin));
        const rotated_imag = real.mul(sin).add(imag.mul(cos));
        return zml.Tensor.stack(&.{ rotated_real, rotated_imag }, .last, .ri)
            .merge(.{ .hd = .{ .pair, .ri } });
    }

    pub fn forward(
        self: ZSingleStreamAttention,
        hidden_states: zml.Tensor,
        attention_mask: ?zml.Tensor,
        freqs_cis: ?zml.Tensor,
    ) zml.Tensor {
        _ = attention_mask;

        var q = self.to_q.forward(hidden_states).splitAxis(.dout, .{ .h = self.heads, .hd = self.head_dim }).rename(.{ .s = .q });
        var k = self.to_k.forward(hidden_states).splitAxis(.dout, .{ .h = self.heads, .hd = self.head_dim }).rename(.{ .s = .k });
        const v = self.to_v.forward(hidden_states).splitAxis(.dout, .{ .h = self.heads, .hd = self.head_dim }).rename(.{ .s = .k });

        if (self.norm_q) |norm| q = norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        if (self.norm_k) |norm| k = norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        if (freqs_cis) |freqs| {
            q = applyFreqs(q, freqs, "q".ptr);
            k = applyFreqs(k, freqs, "k".ptr);
        }

        const attn_out = zml.nn.sdpa(q, k, v, .{})
            .rename(.{ .q = .s })
            .merge(.{ .d = .{ .h, .hd } });
        return self.to_out.forward(attn_out.rename(.{ .d = .dout }));
    }
};

pub const ZImageTransformerBlock = struct {
    dim: u32,
    head_dim: u32,
    attention: ZSingleStreamAttention,
    feed_forward: FeedForward,
    layer_id: u32,
    attention_norm1: RMSNorm,
    ffn_norm1: RMSNorm,
    attention_norm2: RMSNorm,
    ffn_norm2: RMSNorm,
    modulation: bool,
    adaLN_modulation: ?zml.nn.Linear,

    pub fn init(
        store: zml.io.TensorStore.View,
        layer_id: u32,
        dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        norm_eps: f32,
        qk_norm: bool,
        modulation: bool,
    ) !ZImageTransformerBlock {
        _ = n_kv_heads;
        return .{
            .dim = dim,
            .head_dim = @divExact(dim, n_heads),
            .attention = ZSingleStreamAttention.init(store.withPrefix("attention"), dim, n_heads, qk_norm),
            .feed_forward = FeedForward.init(store.withPrefix("feed_forward")),
            .layer_id = layer_id,
            .attention_norm1 = RMSNorm.init(store.withPrefix("attention_norm1"), norm_eps, true, false),
            .ffn_norm1 = RMSNorm.init(store.withPrefix("ffn_norm1"), norm_eps, true, false),
            .attention_norm2 = RMSNorm.init(store.withPrefix("attention_norm2"), norm_eps, true, false),
            .ffn_norm2 = RMSNorm.init(store.withPrefix("ffn_norm2"), norm_eps, true, false),
            .modulation = modulation,
            .adaLN_modulation = if (modulation)
                zml.nn.Linear.init(
                    store.withPrefix("adaLN_modulation").withLayer(0).createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                    store.withPrefix("adaLN_modulation").withLayer(0).createTensor("bias", .{.dout}, .{ .dout = .model }),
                    .d,
                )
            else
                null,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ZImageTransformerBlock)) void {
        ZSingleStreamAttention.unloadBuffers(&self.attention);
        FeedForward.unloadBuffers(&self.feed_forward);
        RMSNorm.unloadBuffers(&self.attention_norm1);
        RMSNorm.unloadBuffers(&self.ffn_norm1);
        RMSNorm.unloadBuffers(&self.attention_norm2);
        RMSNorm.unloadBuffers(&self.ffn_norm2);
        if (self.adaLN_modulation) |*mod| {
            mod.weight.deinit();
            if (mod.bias) |*bias| bias.deinit();
        }
    }

    pub fn forward(
        self: ZImageTransformerBlock,
        x: zml.Tensor,
        attn_mask: ?zml.Tensor,
        freqs_cis: zml.Tensor,
        adaln_input: ?zml.Tensor,
        noise_mask: ?zml.Tensor,
        adaln_noisy: ?zml.Tensor,
        adaln_clean: ?zml.Tensor,
    ) zml.Tensor {
        if (self.modulation) {
            const mod_layer = self.adaLN_modulation orelse @panic("missing adaln modulation");
            const seq_len: u32 = @intCast(x.dim(.s));

            const scale_msa, const gate_msa, const scale_mlp, const gate_mlp = if (noise_mask) |mask| blk: {
                const noisy = mod_layer.forward((adaln_noisy orelse @panic("missing adaln_noisy")).convert(mod_layer.weight.dtype()));
                const clean = mod_layer.forward((adaln_clean orelse @panic("missing adaln_clean")).convert(mod_layer.weight.dtype()));
                const scale_msa_noisy, const gate_msa_noisy, const scale_mlp_noisy, const gate_mlp_noisy = splitModulation(noisy);
                const scale_msa_clean, const gate_msa_clean, const scale_mlp_clean, const gate_mlp_clean = splitModulation(clean);
                break :blk .{
                    selectPerToken(scale_msa_noisy.addConstant(1.0), scale_msa_clean.addConstant(1.0), mask, seq_len),
                    selectPerToken(gate_msa_noisy.tanh(), gate_msa_clean.tanh(), mask, seq_len),
                    selectPerToken(scale_mlp_noisy.addConstant(1.0), scale_mlp_clean.addConstant(1.0), mask, seq_len),
                    selectPerToken(gate_mlp_noisy.tanh(), gate_mlp_clean.tanh(), mask, seq_len),
                };
            } else blk: {
                const mod = mod_layer.forward((adaln_input orelse @panic("missing adaln_input")).convert(mod_layer.weight.dtype()));
                const msa_scale, const msa_gate, const mlp_scale, const mlp_gate = splitModulation(mod);
                const mod_shape = zml.Shape.init(.{ .s = seq_len, .d = msa_scale.dim(.d) }, msa_scale.dtype());
                break :blk .{
                    msa_scale.addConstant(1.0).insertAxes(.d, .{.s}).broad(mod_shape),
                    msa_gate.tanh().insertAxes(.d, .{.s}).broad(mod_shape),
                    mlp_scale.addConstant(1.0).insertAxes(.d, .{.s}).broad(mod_shape),
                    mlp_gate.tanh().insertAxes(.d, .{.s}).broad(mod_shape),
                };
            };

            const attn_in = self.attention_norm1.forward(x).mul(scale_msa);
            const attn_out = self.attention.forward(attn_in, attn_mask, freqs_cis);
            const after_attn = x.add(gate_msa.mul(self.attention_norm2.forward(attn_out)));

            const mlp_in = self.ffn_norm1.forward(after_attn).mul(scale_mlp);
            const mlp_out = self.feed_forward.forward(mlp_in);
            return after_attn.add(gate_mlp.mul(self.ffn_norm2.forward(mlp_out)));
        }

        const attn_out = self.attention.forward(self.attention_norm1.forward(x), attn_mask, freqs_cis);
        const after_attn = x.add(self.attention_norm2.forward(attn_out));
        return after_attn.add(self.ffn_norm2.forward(self.feed_forward.forward(self.ffn_norm1.forward(after_attn))));
    }
};

fn splitModulation(mod: zml.Tensor) struct { zml.Tensor, zml.Tensor, zml.Tensor, zml.Tensor } {
    const scale_msa = mod.slice1d(.dout, .{ .start = 0, .end = @divExact(mod.dim(.dout), 4) }).rename(.{ .dout = .d });
    const gate_msa = mod.slice1d(.dout, .{ .start = @divExact(mod.dim(.dout), 4), .end = @divExact(mod.dim(.dout), 2) }).rename(.{ .dout = .d });
    const scale_mlp = mod.slice1d(.dout, .{ .start = @divExact(mod.dim(.dout), 2), .end = 3 * @divExact(mod.dim(.dout), 4) }).rename(.{ .dout = .d });
    const gate_mlp = mod.slice1d(.dout, .{ .start = 3 * @divExact(mod.dim(.dout), 4), .end = mod.dim(.dout) }).rename(.{ .dout = .d });
    return .{ scale_msa, gate_msa, scale_mlp, gate_mlp };
}
