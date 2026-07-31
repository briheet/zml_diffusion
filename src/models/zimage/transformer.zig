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
        cap_token_count: zml.Tensor,
        cap_aligned_length: zml.Tensor,
        patch_size: u32,
        f_patch_size: u32,
    ) zml.Tensor {
        std.debug.assert(patch_size == self.all_patch_size[0]);
        std.debug.assert(f_patch_size == self.all_f_patch_size[0]);
        std.debug.assert(image.shape().hasTags(.{ .b, .c, .f, .h, .w }));
        std.debug.assert(t.shape().hasTags(.{.b}));
        std.debug.assert(cap_feat.shape().hasTags(.{ .b, .s, .d }));
        std.debug.assert(cap_token_count.shape().hasTags(.{.b}));
        std.debug.assert(cap_aligned_length.shape().hasTags(.{.b}));
        std.debug.assert(image.dim(.b) == t.dim(.b));
        std.debug.assert(image.dim(.b) == cap_feat.dim(.b));
        std.debug.assert(image.dim(.b) == cap_token_count.dim(.b));
        std.debug.assert(image.dim(.b) == cap_aligned_length.dim(.b));
        const size = ImageSize{
            .f = @intCast(image.dim(.f)),
            .h = @intCast(image.dim(.h)),
            .w = @intCast(image.dim(.w)),
        };

        const batch_size: u32 = @intCast(image.dim(.b));
        const timestep_embed = self.t_embedder.forward(t.convert(.f32).scale(self.t_scale));
        const x_patches = patchifyImage(image, patch_size, f_patch_size).convert(self.all_x_embedder[0].weight.dtype());
        const cap_len: u32 = @intCast(cap_feat.dim(.s));
        const padded_cap_len = std.mem.alignForward(u32, cap_len, SEQ_MULTI_OF);
        const cap_masks = preparePromptMasks(batch_size, padded_cap_len, cap_token_count, cap_aligned_length);
        const x_token_count: u32 = @intCast(x_patches.dim(.s));
        const padded_x_token_count = std.mem.alignForward(u32, x_token_count, SEQ_MULTI_OF);

        var x_pos_ids = createCoordinateGrid(
            .{
                @divExact(size.f, f_patch_size),
                @divExact(size.h, patch_size),
                @divExact(size.w, patch_size),
            },
            .{ 0, 0, 0 },
        );
        x_pos_ids = addImagePositionOffsets(x_pos_ids, cap_aligned_length);
        var x_hidden = self.all_x_embedder[0].forward(x_patches).rename(.{ .dout = .d });
        if (padded_x_token_count > x_token_count) {
            const pad_len = padded_x_token_count - x_token_count;
            const pad_shape = zml.Shape.init(.{ .b = batch_size, .s = pad_len, .d = self.dim }, self.x_pad_token.dtype());
            const pad_tokens = self.x_pad_token.rename(.{ .tok = .s }).broad(pad_shape);
            x_hidden = zml.Tensor.concatenate(&.{ x_hidden, pad_tokens }, .s);
            x_pos_ids = zml.Tensor.concatenate(&.{
                x_pos_ids,
                zml.Tensor.zeroes(.init(.{ .b = batch_size, .s = pad_len, .coord = 3 }, .i32)),
            }, .s);
        }
        const x_freqs = self.rope_embedder.forward(x_pos_ids);

        for (self.noise_refiner) |layer| {
            x_hidden = layer.forward(x_hidden, null, x_freqs, timestep_embed, null, null, null);
        }

        var cap_hidden = self.cap_embedder.forward(cap_feat).rename(.{ .dout = .d });
        if (padded_cap_len > cap_len) {
            const pad_shape = zml.Shape.init(.{ .b = batch_size, .s = padded_cap_len - cap_len, .d = self.dim }, self.cap_pad_token.dtype());
            const pad_tokens = self.cap_pad_token.rename(.{ .tok = .s }).broad(pad_shape);
            cap_hidden = zml.Tensor.concatenate(&.{ cap_hidden, pad_tokens }, .s);
        }
        cap_hidden = applyPromptPadding(cap_hidden, cap_masks.tokens, self.cap_pad_token);

        const cap_pos_shape = zml.Shape.init(.{ .b = batch_size, .s = padded_cap_len, .coord = 3 }, .i32);
        const cap_pos_ids = createCoordinateGrid(.{ padded_cap_len, 1, 1 }, .{ 1, 0, 0 })
            .insertAxes(0, .{.b})
            .broad(cap_pos_shape);
        const cap_freqs = self.rope_embedder.forward(cap_pos_ids);

        for (self.context_refiner) |layer| {
            cap_hidden = layer.forward(cap_hidden, cap_masks.aligned, cap_freqs, null, null, null, null);
        }

        var unified = zml.Tensor.concatenate(&.{ x_hidden, cap_hidden }, .s);
        const unified_freqs = zml.Tensor.concatenate(&.{ x_freqs, cap_freqs }, .s);
        const x_mask = zml.Tensor.constant(.{ .bool = true }).broad(
            zml.Shape.init(.{ .b = batch_size, .s = padded_x_token_count }, .bool),
        );
        const unified_mask = zml.Tensor.concatenate(&.{ x_mask, cap_masks.aligned }, .s);

        for (self.layers) |layer| {
            unified = layer.forward(unified, unified_mask, unified_freqs, timestep_embed, null, null, null);
        }

        unified = self.all_final_layer[0].forward(unified, timestep_embed, null, null, null);
        const image_tokens = unified.slice1d(.s, .{ .end = x_patches.dim(.s) });
        return unpatchifyImage(image_tokens, size, patch_size, f_patch_size, self.out_channels);
    }

    pub fn denoiseStep(
        self: *const Transformer,
        latent: zml.Tensor,
        timestep: zml.Tensor,
        prompt_embeds: zml.Tensor,
        prompt_token_count: zml.Tensor,
        prompt_aligned_length: zml.Tensor,
    ) zml.Tensor {
        return self.forward(
            latent.insertAxes(0, .{.b}),
            timestep,
            prompt_embeds.insertAxes(0, .{.b}),
            prompt_token_count.insertAxes(0, .{.b}),
            prompt_aligned_length.insertAxes(0, .{.b}),
            self.all_patch_size[0],
            self.all_f_patch_size[0],
        ).squeeze(.b);
    }

    pub const CfgOutput = struct {
        positive: zml.Tensor,
        negative: zml.Tensor,
    };

    fn splitCfgOutput(output: zml.Tensor) CfgOutput {
        std.debug.assert(output.dim(.b) == 2);
        return .{
            .positive = output.slice1d(.b, .single(0)),
            .negative = output.slice1d(.b, .single(1)),
        };
    }

    const CfgInputs = struct {
        latent: zml.Tensor,
        timestep: zml.Tensor,
        prompt_embeds: zml.Tensor,
        prompt_token_count: zml.Tensor,
        prompt_aligned_length: zml.Tensor,
    };

    fn prepareCfgInputs(
        latent: zml.Tensor,
        timestep: zml.Tensor,
        positive_prompt_embeds: zml.Tensor,
        positive_token_count: zml.Tensor,
        positive_aligned_length: zml.Tensor,
        negative_prompt_embeds: zml.Tensor,
        negative_token_count: zml.Tensor,
        negative_aligned_length: zml.Tensor,
    ) CfgInputs {
        const batch_size = 2;
        return .{
            .latent = zml.Tensor.stack(&.{ latent, latent }, 0, .b),
            .timestep = timestep.broad(zml.Shape.init(.{ .b = batch_size }, timestep.dtype())),
            .prompt_embeds = zml.Tensor.stack(&.{ positive_prompt_embeds, negative_prompt_embeds }, 0, .b),
            .prompt_token_count = zml.Tensor.stack(&.{ positive_token_count, negative_token_count }, 0, .b),
            .prompt_aligned_length = zml.Tensor.stack(&.{ positive_aligned_length, negative_aligned_length }, 0, .b),
        };
    }

    pub fn denoiseCfgStep(
        self: *const Transformer,
        latent: zml.Tensor,
        timestep: zml.Tensor,
        positive_prompt_embeds: zml.Tensor,
        positive_token_count: zml.Tensor,
        positive_aligned_length: zml.Tensor,
        negative_prompt_embeds: zml.Tensor,
        negative_token_count: zml.Tensor,
        negative_aligned_length: zml.Tensor,
    ) CfgOutput {
        const inputs = prepareCfgInputs(
            latent,
            timestep,
            positive_prompt_embeds,
            positive_token_count,
            positive_aligned_length,
            negative_prompt_embeds,
            negative_token_count,
            negative_aligned_length,
        );
        const output = self.forward(
            inputs.latent,
            inputs.timestep,
            inputs.prompt_embeds,
            inputs.prompt_token_count,
            inputs.prompt_aligned_length,
            self.all_patch_size[0],
            self.all_f_patch_size[0],
        );
        return splitCfgOutput(output);
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
        const t_expanded = t.convert(.f32).insertAxes(.last, .{.f}).broad(shape);
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
            const scale_shape = zml.Shape.init(.{ .b = x.dim(.b), .s = seq_len, .d = global.dim(.d) }, global.dtype());
            break :blk global.insertAxes(.d, .{.s}).broad(scale_shape);
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
        std.debug.assert(ids.rank() == 2 or ids.rank() == 3);
        std.debug.assert(ids.shape().hasTags(.{ .s, .coord }));

        var result: [3]zml.Tensor = undefined;
        inline for (0..3) |i| {
            const axis_ids = ids.slice1d(-1, .single(i));
            const half = @divFloor(self.axes_dims[i], 2);
            const freqs = zml.Tensor.arange(.{ .end = half }, .f32)
                .withTags(.{.hd})
                .scale(-2.0 * std.math.log(f32, std.math.e, self.theta) / @as(f32, @floatFromInt(self.axes_dims[i])))
                .exp();
            result[i] = if (ids.rank() == 2)
                axis_ids.convert(.f32).outer(freqs)
            else blk: {
                const phase_shape = zml.Shape.init(.{
                    .b = ids.dim(.b),
                    .s = ids.dim(.s),
                    .hd = half,
                }, .f32);
                const expanded_ids = axis_ids.convert(.f32).insertAxes(.last, .{.hd}).broad(phase_shape);
                const expanded_freqs = freqs.insertAxes(.hd, .{ .b, .s }).broad(phase_shape);
                break :blk expanded_ids.mul(expanded_freqs);
            };
        }

        return zml.Tensor.concatenate(&result, -1);
    }
};

const ImageSize = struct {
    f: u32,
    h: u32,
    w: u32,
};

const PromptMasks = struct {
    tokens: zml.Tensor,
    aligned: zml.Tensor,
};

fn preparePromptMasks(
    batch_size: u32,
    capacity: u32,
    token_count: zml.Tensor,
    aligned_length: zml.Tensor,
) PromptMasks {
    const shape = zml.Shape.init(.{ .b = batch_size, .s = capacity }, .i32);
    const positions = zml.Tensor.iota(shape, .s);
    const token_limits = token_count.convert(.i32).insertAxes(.last, .{.s}).broad(shape);
    const aligned_limits = aligned_length.convert(.i32).insertAxes(.last, .{.s}).broad(shape);
    return .{
        .tokens = positions.cmp(.LT, token_limits),
        .aligned = positions.cmp(.LT, aligned_limits),
    };
}

fn applyPromptPadding(hidden: zml.Tensor, token_mask: zml.Tensor, pad_token: zml.Tensor) zml.Tensor {
    const pad_tokens = pad_token.rename(.{ .tok = .s }).broad(hidden.shape());
    const expanded_token_mask = token_mask.insertAxes(.last, .{.d}).broad(hidden.shape());
    return expanded_token_mask.select(hidden, pad_tokens);
}

fn addImagePositionOffsets(position_ids: zml.Tensor, aligned_length: zml.Tensor) zml.Tensor {
    std.debug.assert(position_ids.shape().hasTags(.{ .s, .coord }));
    std.debug.assert(aligned_length.shape().hasTags(.{.b}));

    const batch_size: u32 = @intCast(aligned_length.dim(.b));
    const position_shape = zml.Shape.init(.{
        .b = batch_size,
        .s = position_ids.dim(.s),
        .coord = position_ids.dim(.coord),
    }, position_ids.dtype());
    const batched_positions = position_ids.insertAxes(0, .{.b}).broad(position_shape);
    const temporal_offset = aligned_length.convert(position_ids.dtype()).addConstant(1);
    const zero_offset = zml.Tensor.zeroes(temporal_offset.shape());
    const offsets = zml.Tensor.stack(&.{ temporal_offset, zero_offset, zero_offset }, .last, .coord)
        .insertAxes(.coord, .{.s})
        .broad(position_shape);
    return batched_positions.add(offsets);
}

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
        .transpose(.{ .b, .ft, .ht, .wt, .pf, .ph, .pw, .c })
        .merge(.{ .s = .{ .ft, .ht, .wt }, .d = .{ .pf, .ph, .pw, .c } });
}

fn unpatchifyImage(
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
        .transpose(.{ .b, .c, .ft, .pf, .ht, .ph, .wt, .pw })
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
        var q = self.to_q.forward(hidden_states).splitAxis(.dout, .{ .h = self.heads, .hd = self.head_dim }).rename(.{ .s = .q });
        var k = self.to_k.forward(hidden_states).splitAxis(.dout, .{ .h = self.heads, .hd = self.head_dim }).rename(.{ .s = .k });
        const v = self.to_v.forward(hidden_states).splitAxis(.dout, .{ .h = self.heads, .hd = self.head_dim }).rename(.{ .s = .k });

        if (self.norm_q) |norm| q = norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        if (self.norm_k) |norm| k = norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        if (freqs_cis) |freqs| {
            q = applyFreqs(q, freqs, "q".ptr);
            k = applyFreqs(k, freqs, "k".ptr);
        }

        const additive_mask = if (attention_mask) |mask| prepareAttentionMask(mask, q.dtype()) else null;

        const attn_out = zml.nn.sdpa(q, k, v, .{ .attn_mask = additive_mask })
            .rename(.{ .q = .s })
            .merge(.{ .d = .{ .h, .hd } });
        return self.to_out.forward(attn_out.rename(.{ .d = .dout }));
    }
};

fn prepareAttentionMask(mask: zml.Tensor, dtype: zml.DataType) zml.Tensor {
    const key_mask = mask.rename(.{ .s = .k });
    const zeros = zml.Tensor.constant(dtype.zero()).broad(key_mask.shape());
    const masked = zml.Tensor.constant(dtype.minValue()).broad(key_mask.shape());
    return key_mask.select(zeros, masked);
}

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
                const mod_shape = zml.Shape.init(.{ .b = x.dim(.b), .s = seq_len, .d = msa_scale.dim(.d) }, msa_scale.dtype());
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

test "CFG inputs repeat latent and preserve prompt order" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    const latent: zml.Tensor = .init(.{ .c = 1, .f = 1, .h = 1, .w = 2 }, .f32);
    const timestep: zml.Tensor = .init(.{ .b = 1 }, .f32);
    const positive_prompt: zml.Tensor = .init(.{ .s = 2, .d = 1 }, .f32);
    const negative_prompt: zml.Tensor = .init(.{ .s = 2, .d = 1 }, .f32);
    const length_shape = zml.Shape.init(.{}, .u32);
    const positive_token_count: zml.Tensor = .fromShape(length_shape);
    const positive_aligned_length: zml.Tensor = .fromShape(length_shape);
    const negative_token_count: zml.Tensor = .fromShape(length_shape);
    const negative_aligned_length: zml.Tensor = .fromShape(length_shape);

    const forward = struct {
        fn forward(
            latent_: zml.Tensor,
            timestep_: zml.Tensor,
            positive_: zml.Tensor,
            positive_token_count_: zml.Tensor,
            positive_aligned_length_: zml.Tensor,
            negative_: zml.Tensor,
            negative_token_count_: zml.Tensor,
            negative_aligned_length_: zml.Tensor,
        ) Transformer.CfgInputs {
            return Transformer.prepareCfgInputs(
                latent_,
                timestep_,
                positive_,
                positive_token_count_,
                positive_aligned_length_,
                negative_,
                negative_token_count_,
                negative_aligned_length_,
            );
        }
    }.forward;

    var exe = try zml.module.compile(
        allocator,
        io,
        forward,
        .{
            latent,
            timestep,
            positive_prompt,
            positive_token_count,
            positive_aligned_length,
            negative_prompt,
            negative_token_count,
            negative_aligned_length,
        },
        platform,
        .{},
    );
    defer exe.deinit();

    var latent_buffer = try zml.Buffer.fromBytes(io, platform, latent.shape(), .replicated, std.mem.sliceAsBytes(&[_]f32{ 1, 2 }));
    defer latent_buffer.deinit();
    var timestep_buffer = try zml.Buffer.fromBytes(io, platform, timestep.shape(), .replicated, std.mem.sliceAsBytes(&[_]f32{0.25}));
    defer timestep_buffer.deinit();
    var positive_buffer = try zml.Buffer.fromBytes(io, platform, positive_prompt.shape(), .replicated, std.mem.sliceAsBytes(&[_]f32{ 10, 11 }));
    defer positive_buffer.deinit();
    var negative_buffer = try zml.Buffer.fromBytes(io, platform, negative_prompt.shape(), .replicated, std.mem.sliceAsBytes(&[_]f32{ 20, 21 }));
    defer negative_buffer.deinit();
    var positive_token_count_buffer = try zml.Buffer.scalar(io, platform, @as(u32, 35), .u32);
    defer positive_token_count_buffer.deinit();
    var positive_aligned_length_buffer = try zml.Buffer.scalar(io, platform, @as(u32, 64), .u32);
    defer positive_aligned_length_buffer.deinit();
    var negative_token_count_buffer = try zml.Buffer.scalar(io, platform, @as(u32, 7), .u32);
    defer negative_token_count_buffer.deinit();
    var negative_aligned_length_buffer = try zml.Buffer.scalar(io, platform, @as(u32, 32), .u32);
    defer negative_aligned_length_buffer.deinit();

    var result = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        forward,
        .{
            latent_buffer,
            timestep_buffer,
            positive_buffer,
            positive_token_count_buffer,
            positive_aligned_length_buffer,
            negative_buffer,
            negative_token_count_buffer,
            negative_aligned_length_buffer,
        },
    );
    defer result.latent.deinit();
    defer result.timestep.deinit();
    defer result.prompt_embeds.deinit();
    defer result.prompt_token_count.deinit();
    defer result.prompt_aligned_length.deinit();

    var latent_slice = try result.latent.toSliceAlloc(allocator, io);
    defer latent_slice.free(allocator);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 1, 2 }, latent_slice.constItems(f32));

    var timestep_slice = try result.timestep.toSliceAlloc(allocator, io);
    defer timestep_slice.free(allocator);
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.25 }, timestep_slice.constItems(f32));

    var prompt_slice = try result.prompt_embeds.toSliceAlloc(allocator, io);
    defer prompt_slice.free(allocator);
    try std.testing.expectEqualSlices(f32, &.{ 10, 11, 20, 21 }, prompt_slice.constItems(f32));

    var token_count_slice = try result.prompt_token_count.toSliceAlloc(allocator, io);
    defer token_count_slice.free(allocator);
    try std.testing.expectEqualSlices(u32, &.{ 35, 7 }, token_count_slice.constItems(u32));

    var aligned_length_slice = try result.prompt_aligned_length.toSliceAlloc(allocator, io);
    defer aligned_length_slice.free(allocator);
    try std.testing.expectEqualSlices(u32, &.{ 64, 32 }, aligned_length_slice.constItems(u32));
}

test "CFG output split removes the batch dimension" {
    const output: zml.Tensor = .init(.{ .b = 2, .c = 16, .f = 1, .h = 128, .w = 128 }, .bf16);
    const split = Transformer.splitCfgOutput(output);

    try std.testing.expect(!split.positive.shape().hasTags(.{.b}));
    try std.testing.expect(!split.negative.shape().hasTags(.{.b}));
    try std.testing.expectEqual(@as(i64, 16), split.positive.dim(.c));
    try std.testing.expectEqual(@as(i64, 16), split.negative.dim(.c));
}

test "Diffusers prompt masks and image RoPE offsets vary per CFG item" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    const token_count: zml.Tensor = .init(.{ .b = 2 }, .u32);
    const aligned_length: zml.Tensor = .init(.{ .b = 2 }, .u32);
    const Output = struct {
        tokens: zml.Tensor,
        aligned: zml.Tensor,
        frequencies: zml.Tensor,
    };
    const forward = struct {
        fn forward(token_count_: zml.Tensor, aligned_length_: zml.Tensor) Output {
            const masks = preparePromptMasks(2, 96, token_count_, aligned_length_);
            const positions = addImagePositionOffsets(
                createCoordinateGrid(.{ 1, 1, 2 }, .{ 0, 0, 0 }),
                aligned_length_,
            );
            const rope: RopeEmbedder = .{
                .theta = 256.0,
                .axes_dims = .{ 2, 2, 2 },
                .axes_lens = .{ 128, 128, 128 },
            };
            return .{
                .tokens = masks.tokens,
                .aligned = masks.aligned,
                .frequencies = rope.forward(positions),
            };
        }
    }.forward;

    var exe = try zml.module.compile(
        allocator,
        io,
        forward,
        .{ token_count, aligned_length },
        platform,
        .{},
    );
    defer exe.deinit();

    var token_count_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        token_count.shape(),
        .replicated,
        std.mem.sliceAsBytes(&[_]u32{ 35, 7 }),
    );
    defer token_count_buffer.deinit();
    var aligned_length_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        aligned_length.shape(),
        .replicated,
        std.mem.sliceAsBytes(&[_]u32{ 64, 32 }),
    );
    defer aligned_length_buffer.deinit();

    var result = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        forward,
        .{ token_count_buffer, aligned_length_buffer },
    );
    defer result.tokens.deinit();
    defer result.aligned.deinit();
    defer result.frequencies.deinit();

    var token_mask_slice = try result.tokens.toSliceAlloc(allocator, io);
    defer token_mask_slice.free(allocator);
    const token_mask = token_mask_slice.constItems(bool);
    try std.testing.expectEqual(@as(usize, 35), std.mem.count(bool, token_mask[0..96], &.{true}));
    try std.testing.expectEqual(@as(usize, 7), std.mem.count(bool, token_mask[96..192], &.{true}));

    var aligned_mask_slice = try result.aligned.toSliceAlloc(allocator, io);
    defer aligned_mask_slice.free(allocator);
    const aligned_mask = aligned_mask_slice.constItems(bool);
    try std.testing.expectEqual(@as(usize, 64), std.mem.count(bool, aligned_mask[0..96], &.{true}));
    try std.testing.expectEqual(@as(usize, 32), std.mem.count(bool, aligned_mask[96..192], &.{true}));

    var frequencies_slice = try result.frequencies.toSliceAlloc(allocator, io);
    defer frequencies_slice.free(allocator);
    try std.testing.expectEqualSlices(
        f32,
        &.{ 65, 0, 0, 65, 0, 1, 33, 0, 0, 33, 0, 1 },
        frequencies_slice.constItems(f32),
    );
}

test "masked caption keys do not affect attention output" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    const q: zml.Tensor = .init(.{ .b = 1, .h = 1, .q = 1, .hd = 1 }, .f32);
    const k: zml.Tensor = .init(.{ .b = 1, .h = 1, .k = 2, .hd = 1 }, .f32);
    const v: zml.Tensor = .init(.{ .b = 1, .h = 1, .k = 2, .hd = 1 }, .f32);
    const mask: zml.Tensor = .init(.{ .b = 1, .s = 2 }, .bool);
    const forward = struct {
        fn forward(q_: zml.Tensor, k_: zml.Tensor, v_: zml.Tensor, mask_: zml.Tensor) zml.Tensor {
            return zml.nn.sdpa(q_, k_, v_, .{ .attn_mask = prepareAttentionMask(mask_, q_.dtype()) });
        }
    }.forward;

    var exe = try zml.module.compile(allocator, io, forward, .{ q, k, v, mask }, platform, .{});
    defer exe.deinit();

    var q_buffer = try zml.Buffer.fromBytes(io, platform, q.shape(), .replicated, std.mem.sliceAsBytes(&[_]f32{0}));
    defer q_buffer.deinit();
    var k_buffer = try zml.Buffer.fromBytes(io, platform, k.shape(), .replicated, std.mem.sliceAsBytes(&[_]f32{ 0, 0 }));
    defer k_buffer.deinit();
    var v_buffer = try zml.Buffer.fromBytes(io, platform, v.shape(), .replicated, std.mem.sliceAsBytes(&[_]f32{ 2, 1000 }));
    defer v_buffer.deinit();
    var mask_buffer = try zml.Buffer.fromBytes(io, platform, mask.shape(), .replicated, std.mem.sliceAsBytes(&[_]bool{ true, false }));
    defer mask_buffer.deinit();

    var result = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        forward,
        .{ q_buffer, k_buffer, v_buffer, mask_buffer },
    );
    defer result.deinit();

    var result_slice = try result.toSliceAlloc(allocator, io);
    defer result_slice.free(allocator);
    try std.testing.expectEqualSlices(f32, &.{2}, result_slice.constItems(f32));
}

test "Qwen padding is replaced with the learned caption pad token" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    const hidden: zml.Tensor = .init(.{ .b = 2, .s = 4, .d = 1 }, .f32);
    const token_mask: zml.Tensor = .init(.{ .b = 2, .s = 4 }, .bool);
    const pad_token: zml.Tensor = .init(.{ .tok = 1, .d = 1 }, .f32);

    var exe = try zml.module.compile(
        allocator,
        io,
        applyPromptPadding,
        .{ hidden, token_mask, pad_token },
        platform,
        .{},
    );
    defer exe.deinit();

    var hidden_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        hidden.shape(),
        .replicated,
        std.mem.sliceAsBytes(&[_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 }),
    );
    defer hidden_buffer.deinit();
    var mask_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        token_mask.shape(),
        .replicated,
        std.mem.sliceAsBytes(&[_]bool{ true, true, true, false, true, false, false, false }),
    );
    defer mask_buffer.deinit();
    var pad_buffer = try zml.Buffer.fromBytes(
        io,
        platform,
        pad_token.shape(),
        .replicated,
        std.mem.sliceAsBytes(&[_]f32{99}),
    );
    defer pad_buffer.deinit();

    var result = try zml.testing.autoCall(
        allocator,
        io,
        &exe,
        applyPromptPadding,
        .{ hidden_buffer, mask_buffer, pad_buffer },
    );
    defer result.deinit();

    var result_slice = try result.toSliceAlloc(allocator, io);
    defer result_slice.free(allocator);
    try std.testing.expectEqualSlices(
        f32,
        &.{ 1, 2, 3, 99, 5, 99, 99, 99 },
        result_slice.constItems(f32),
    );
}

test "batched patchify and unpatchify round trip" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    const input: zml.Tensor = .init(.{ .b = 2, .c = 2, .f = 1, .h = 2, .w = 2 }, .f32);
    const forward = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            const size: ImageSize = .{ .f = 1, .h = 2, .w = 2 };
            return unpatchifyImage(patchifyImage(x, 2, 1), size, 2, 1, 2);
        }
    }.forward;

    var exe = try zml.module.compile(allocator, io, forward, .{input}, platform, .{});
    defer exe.deinit();

    var input_values: [16]f32 = undefined;
    for (&input_values, 0..) |*value, i| value.* = @floatFromInt(i);

    var input_buffer = try zml.Buffer.fromBytes(io, platform, input.shape(), .replicated, std.mem.sliceAsBytes(&input_values));
    defer input_buffer.deinit();

    var result = try zml.testing.autoCall(allocator, io, &exe, forward, .{input_buffer});
    defer result.deinit();

    var result_slice = try result.toSliceAlloc(allocator, io);
    defer result_slice.free(allocator);
    try std.testing.expectEqualSlices(f32, &input_values, result_slice.constItems(f32));
}

test "RoPE frequencies broadcast across CFG batch" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();

    const input: zml.Tensor = .init(.{ .b = 2, .q = 3, .h = 2, .hd = 4 }, .f32);
    const frequencies: zml.Tensor = .init(.{ .s = 3, .hd = 2 }, .f32);
    const forward = struct {
        fn forward(x: zml.Tensor, freqs: zml.Tensor) zml.Tensor {
            return ZSingleStreamAttention.applyFreqs(x, freqs, "q".ptr);
        }
    }.forward;

    var exe = try zml.module.compile(allocator, io, forward, .{ input, frequencies }, platform, .{});
    defer exe.deinit();

    var input_values: [48]f32 = undefined;
    for (&input_values, 0..) |*value, i| value.* = @floatFromInt(i);
    const frequency_values = [_]f32{0} ** 6;

    var input_buffer = try zml.Buffer.fromBytes(io, platform, input.shape(), .replicated, std.mem.sliceAsBytes(&input_values));
    defer input_buffer.deinit();
    var frequency_buffer = try zml.Buffer.fromBytes(io, platform, frequencies.shape(), .replicated, std.mem.sliceAsBytes(&frequency_values));
    defer frequency_buffer.deinit();

    var result = try zml.testing.autoCall(allocator, io, &exe, forward, .{ input_buffer, frequency_buffer });
    defer result.deinit();

    var result_slice = try result.toSliceAlloc(allocator, io);
    defer result_slice.free(allocator);
    try std.testing.expectEqualSlices(f32, &input_values, result_slice.constItems(f32));
}
