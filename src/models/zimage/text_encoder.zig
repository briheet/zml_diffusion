const std = @import("std");

const zml = @import("zml");

pub const LayerType = enum {
    full_attention,
    sliding_attention,
};

pub const TextEncoder = struct {
    // Config from https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/text_encoder/config.json
    pub const Config = struct {
        architecture: [1][]const u8 = .{"Qwen3ForCausalLM"},
        attention_bias: bool = false,
        attention_dropout: f32 = 0.0,
        bos_token_id: u32 = 151643,
        eos_token_id: u32 = 151645,
        head_dim: u32 = 128,
        hidden_act: []const u8 = "silu",
        hidden_size: u32 = 2560,
        initializer_range: f32 = 0.02,
        intermediate_size: u32 = 9728,
        max_position_embeddings: u32 = 40960,
        num_hidden_layers: u32 = 36,
        max_window_layers: u32 = 36,
        model_type: []const u8 = "qwen3",
        num_attention_heads: u32 = 32,
        num_key_value_heads: u32 = 8,
        pad_token_id: ?u32 = null,
        rms_norm_eps: f32 = 1e-6,
        rope_scaling: ?u32 = null,
        rope_theta: f32 = 1000000.0,
        sliding_window: ?u32 = null,
        tie_word_embeddings: bool = true,
        torch_dtype: []const u8 = "bfloat16",
        transformers_version: []const u8 = "4.51.0",
        use_cache: bool = true,
        use_sliding_window: bool = false,
        vocab_size: u32 = 151936,
        layer_types: ?[]const []const u8 = null,
    };

    inner: Qwen3ForCausalLM,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !TextEncoder {
        return .{
            .inner = try Qwen3ForCausalLM.init(allocator, store, config),
            .config = config,
        };
    }

    pub fn deinit(self: *TextEncoder, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
    }
};

pub const Qwen3ForCausalLM = struct {
    model: Qwen3Model,
    vocab_size: u32,
    lm_head: ?zml.nn.Linear,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: TextEncoder.Config) !Qwen3ForCausalLM {
        return .{
            .model = try Qwen3Model.init(allocator, store.withPrefix("model"), config),
            .vocab_size = config.vocab_size,
            .lm_head = if (store.hasKey("lm_head.weight"))
                .init(
                    store.withPrefix("lm_head").createTensor("weight", .{ .voc, .d }, .{
                        .voc = .replicated,
                        .d = .model,
                    }),
                    null,
                    .d,
                )
            else
                null,
        };
    }

    pub fn deinit(self: Qwen3ForCausalLM, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
    }

    pub fn forward(self: Qwen3ForCausalLM, input_ids: zml.Tensor) zml.Tensor {
        const hidden_states = self.model.forward(input_ids, null, null);
        return (self.lm_head orelse @panic("missing lm_head")).forward(hidden_states);
    }

    pub fn forwardHidden(self: Qwen3ForCausalLM, input_ids: zml.Tensor, attention_mask: ?zml.Tensor) zml.Tensor {
        return self.model.forward(input_ids, attention_mask, null);
    }

    pub fn encodePrompt(self: Qwen3ForCausalLM, input_ids: zml.Tensor, attention_mask: zml.Tensor) zml.Tensor {
        // Diffusers requests hidden_states[-2], which is the input to Qwen3's
        // final decoder layer rather than the final normalized model output.
        return self.model.forwardThroughLayers(
            input_ids,
            attention_mask,
            null,
            self.model.layers.len - 1,
        ).squeeze(.b).convert(.f32);
    }
};

pub const Qwen3Model = struct {
    padding_idx: ?u32,
    vocab_size: u32,
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []Qwen3DecoderLayer,
    norm: Qwen3RMSNorm,
    rotary_emb: Qwen3RotaryEmbedding,
    gradient_checkpointing: bool,
    has_sliding_layers: bool,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: TextEncoder.Config,
    ) !Qwen3Model {
        const layers = try allocator.alloc(Qwen3DecoderLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = .init(
                store.withPrefix("layers").withLayer(i),
                config,
                @intCast(i),
                layerTypeAt(config, i),
            );
        }

        return .{
            .padding_idx = config.pad_token_id,
            .vocab_size = config.vocab_size,
            .embed_tokens = .{
                .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, .{
                    .voc = .replicated,
                    .d = .model,
                }),
            },
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), config.rms_norm_eps),
            .rotary_emb = .init(config),
            .gradient_checkpointing = false,
            .has_sliding_layers = hasSlidingLayers(config),
        };
    }

    pub fn deinit(self: Qwen3Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Qwen3Model), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            Qwen3DecoderLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        self.norm.weight.deinit();
    }

    pub fn forward(
        self: Qwen3Model,
        input_ids: zml.Tensor,
        attention_mask: ?zml.Tensor,
        position_ids: ?zml.Tensor,
    ) zml.Tensor {
        return self.norm.forward(self.forwardPreNorm(input_ids, attention_mask, position_ids));
    }

    pub fn forwardPreNorm(
        self: Qwen3Model,
        input_ids: zml.Tensor,
        attention_mask: ?zml.Tensor,
        position_ids: ?zml.Tensor,
    ) zml.Tensor {
        return self.forwardThroughLayers(input_ids, attention_mask, position_ids, self.layers.len);
    }

    pub fn forwardThroughLayers(
        self: Qwen3Model,
        input_ids: zml.Tensor,
        attention_mask: ?zml.Tensor,
        position_ids: ?zml.Tensor,
        layer_count: usize,
    ) zml.Tensor {
        std.debug.assert(layer_count <= self.layers.len);
        var hidden_states = self.embed_tokens.weight.gather(.{ .voc = input_ids }, .{});
        const resolved_position_ids = position_ids orelse zml.Tensor.arange(.{ .end = hidden_states.dim(.s) }, .i64)
            .withTags(.{.s})
            .insertAxes(.s, .{.b})
            .broad(zml.Shape.init(.{ .b = hidden_states.dim(.b), .s = hidden_states.dim(.s) }, .i64));
        const position_embeddings = self.rotary_emb.forward(hidden_states, resolved_position_ids);

        for (self.layers[0..layer_count]) |layer| {
            hidden_states = layer.forward(
                hidden_states,
                attention_mask,
                position_embeddings,
            );
        }

        return hidden_states;
    }
};

fn layerTypeAt(config: TextEncoder.Config, layer_idx: usize) ?[]const u8 {
    if (config.layer_types) |layer_types| return layer_types[layer_idx];
    if (!config.use_sliding_window or config.sliding_window == null) return "full_attention";
    return if (layer_idx >= config.max_window_layers) "sliding_attention" else "full_attention";
}

fn hasSlidingLayers(config: TextEncoder.Config) bool {
    if (config.layer_types) |layer_types| {
        for (layer_types) |layer_type| {
            if (std.mem.eql(u8, layer_type, "sliding_attention")) return true;
        }
        return false;
    }
    return config.use_sliding_window and config.sliding_window != null and config.max_window_layers < config.num_hidden_layers;
}

pub const Qwen3DecoderLayer = struct {
    hidden_size: u32,
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: Qwen3RMSNorm,
    post_attention_layernorm: Qwen3RMSNorm,

    pub fn init(
        store: zml.io.TensorStore.View,
        config: TextEncoder.Config,
        layer_idx: u32,
        layer_type: ?[]const u8,
    ) Qwen3DecoderLayer {
        return .{
            .hidden_size = config.hidden_size,
            .self_attn = .init(store.withPrefix("self_attn"), config, layer_idx, layer_type),
            .mlp = .init(store.withPrefix("mlp"), config),
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Qwen3DecoderLayer)) void {
        Qwen3Attention.unloadBuffers(&self.self_attn);
        Qwen3MLP.unloadBuffers(&self.mlp);
        self.input_layernorm.weight.deinit();
        self.post_attention_layernorm.weight.deinit();
    }

    pub fn forward(
        self: Qwen3DecoderLayer,
        hidden_states: zml.Tensor,
        attention_mask: ?zml.Tensor,
        position_embeddings: struct { zml.Tensor, zml.Tensor },
    ) zml.Tensor {
        const residual1 = hidden_states;
        const normalized1 = self.input_layernorm.forward(hidden_states);

        const attn_out = self.self_attn.forward(
            normalized1,
            position_embeddings,
            attention_mask,
        );
        const after_attn = residual1.add(attn_out);

        const residual2 = after_attn;
        const normalized2 = self.post_attention_layernorm.forward(after_attn);
        const mlp_out = self.mlp.forward(normalized2);

        return residual2.add(mlp_out);
    }
};

pub const Qwen3RMSNorm = struct {
    weight: zml.Tensor,
    variance_epsilon: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) Qwen3RMSNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .{ .d = .replicated }),
            .variance_epsilon = eps,
        };
    }

    pub fn forward(self: Qwen3RMSNorm, x: zml.Tensor) zml.Tensor {
        const x_f32 = x.convert(.f32);
        const weight_f32 = self.weight.convert(.f32);

        const normalized = zml.nn.rmsNorm(x_f32, .d, self.variance_epsilon);
        return normalized.mul(weight_f32.broad(x.shape())).convert(x.dtype());
    }
};

pub const Qwen3RotaryEmbedding = struct {
    max_seq_len_cached: u32,
    original_max_seq_len: u32,
    rope_opts: zml.nn.RopeOpts,
    rotary_dim: u32,

    pub fn init(config: TextEncoder.Config) Qwen3RotaryEmbedding {
        return .{
            .max_seq_len_cached = config.max_position_embeddings,
            .original_max_seq_len = config.max_position_embeddings,
            .rope_opts = .{
                .layout = .real_im_pass,
                .scaling = .{ .default = .{ .rope_theta = config.rope_theta } },
            },
            .rotary_dim = config.head_dim,
        };
    }

    pub fn forward(
        self: Qwen3RotaryEmbedding,
        x: zml.Tensor,
        position_ids: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor } {
        const inv_freq = zml.nn.invFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});
        const freqs = position_ids.convert(.f32).outer(inv_freq);
        const emb = zml.Tensor.concatenate(&.{ freqs, freqs }, -1);
        const cos = emb.cos().convert(x.dtype());
        const sin = emb.sin().convert(x.dtype());

        return .{ cos, sin };
    }
};

pub const Qwen3Attention = struct {
    layer_type: ?[]const u8 = null,
    config: TextEncoder.Config,
    layer_idx: u32,
    head_dim: u32,
    num_key_value_groups: u32,
    scaling: f32,
    attention_dropout: f32,
    is_causal: bool,

    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    o_proj: zml.nn.Linear,

    q_norm: Qwen3RMSNorm,
    k_norm: Qwen3RMSNorm,

    sliding_window: ?u32,

    pub fn init(
        store: zml.io.TensorStore.View,
        config: TextEncoder.Config,
        layer_idx: u32,
        layer_type: ?[]const u8,
    ) Qwen3Attention {
        const head_dim = config.head_dim;
        const num_key_value_groups = @divExact(config.num_attention_heads, config.num_key_value_heads);

        return .{
            .layer_type = layer_type,
            .config = config,
            .layer_idx = layer_idx,
            .head_dim = head_dim,
            .num_key_value_groups = num_key_value_groups,
            .scaling = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim))),
            .attention_dropout = config.attention_dropout,
            .is_causal = true,

            .q_proj = .init(
                store.withPrefix("q_proj").createTensor("weight", .{ .dout, .d }, .{
                    .dout = .model,
                    .d = .replicated,
                }),
                if (config.attention_bias)
                    store.withPrefix("q_proj").createTensor("bias", .{.dout}, .{ .dout = .model })
                else
                    null,
                .d,
            ),
            .k_proj = .init(
                store.withPrefix("k_proj").createTensor("weight", .{ .dout, .d }, .{
                    .dout = .model,
                    .d = .replicated,
                }),
                if (config.attention_bias)
                    store.withPrefix("k_proj").createTensor("bias", .{.dout}, .{ .dout = .model })
                else
                    null,
                .d,
            ),
            .v_proj = .init(
                store.withPrefix("v_proj").createTensor("weight", .{ .dout, .d }, .{
                    .dout = .model,
                    .d = .replicated,
                }),
                if (config.attention_bias)
                    store.withPrefix("v_proj").createTensor("bias", .{.dout}, .{ .dout = .model })
                else
                    null,
                .d,
            ),
            .o_proj = .init(
                store.withPrefix("o_proj").createTensor("weight", .{ .d, .dout }, .{
                    .d = .replicated,
                    .dout = .model,
                }),
                if (config.attention_bias)
                    store.withPrefix("o_proj").createTensor("bias", .{.d}, .{ .d = .replicated })
                else
                    null,
                .dout,
            ),

            .q_norm = .init(store.withPrefix("q_norm"), config.rms_norm_eps),
            .k_norm = .init(store.withPrefix("k_norm"), config.rms_norm_eps),

            .sliding_window = if (layer_type != null and std.mem.eql(u8, layer_type.?, "sliding_attention"))
                config.sliding_window
            else
                null,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Qwen3Attention)) void {
        self.q_proj.weight.deinit();
        if (self.q_proj.bias) |*b| b.deinit();
        self.k_proj.weight.deinit();
        if (self.k_proj.bias) |*b| b.deinit();
        self.v_proj.weight.deinit();
        if (self.v_proj.bias) |*b| b.deinit();
        self.o_proj.weight.deinit();
        if (self.o_proj.bias) |*b| b.deinit();
        self.q_norm.weight.deinit();
        self.k_norm.weight.deinit();
    }

    pub fn forward(
        self: Qwen3Attention,
        hidden_states: zml.Tensor,
        position_embeddings: struct { zml.Tensor, zml.Tensor },
        attention_mask: ?zml.Tensor,
    ) zml.Tensor {
        const cos, const sin = position_embeddings;

        var query_states = self.q_proj.forward(hidden_states);
        var key_states = self.k_proj.forward(hidden_states);
        var value_states = self.v_proj.forward(hidden_states);

        query_states = query_states.splitAxis(.dout, .{ .h = self.config.num_attention_heads, .hd = self.head_dim });
        key_states = key_states.splitAxis(.dout, .{ .hk = self.config.num_key_value_heads, .hd = self.head_dim });
        value_states = value_states.splitAxis(.dout, .{ .hk = self.config.num_key_value_heads, .hd = self.head_dim });

        query_states = self.q_norm.forward(query_states.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        key_states = self.k_norm.forward(key_states.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        query_states, key_states = applyRotaryPosEmb(query_states, key_states, cos, sin);

        const q = query_states.rename(.{ .s = .q });
        const k = repeatKv(key_states, self.num_key_value_groups).rename(.{ .s = .k });
        const v = repeatKv(value_states, self.num_key_value_groups).rename(.{ .s = .k });
        var attn_mask = zml.nn.causalAttnMask(.{ .q = q.dim(.q), .k = k.dim(.k) }, q.dtype(), self.sliding_window);
        if (attention_mask) |mask| {
            const key_mask = mask.squeeze(.b)
                .rename(.{ .s = .k })
                .insertAxes(.k, .{.q})
                .broad(zml.Shape.init(.{ .q = q.dim(.q), .k = k.dim(.k) }, .bool));
            const zeros = zml.Tensor.constant(q.dtype().zero()).broad(key_mask.shape());
            const minus_inf = zml.Tensor.constant(q.dtype().minValue()).broad(key_mask.shape());
            const padding_mask = zml.Tensor.select(key_mask, zeros, minus_inf);
            attn_mask = attn_mask.add(padding_mask);
        }
        const attn_output = zml.nn.sdpa(q.squeeze(.b), k.squeeze(.b), v.squeeze(.b), .{
            .attn_mask = attn_mask,
            .scale = zml.Tensor.scalar(self.scaling, .f32),
        })
            .insertAxes(.q, .{.b})
            .rename(.{ .q = .s })
            .merge(.{ .dout = .{ .h, .hd } });

        return self.o_proj.forward(attn_output);
    }

    pub fn applyRotaryPosEmb(
        q: zml.Tensor,
        k: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    ) struct { zml.Tensor, zml.Tensor } {
        const q_cos = cos.insertAxes(.hd, .{.h}).broad(q.shape());
        const q_sin = sin.insertAxes(.hd, .{.h}).broad(q.shape());
        const k_cos = cos.insertAxes(.hd, .{.hk}).broad(k.shape());
        const k_sin = sin.insertAxes(.hd, .{.hk}).broad(k.shape());

        const q_embed = q.mul(q_cos).add(rotateHalf(q).mul(q_sin));
        const k_embed = k.mul(k_cos).add(rotateHalf(k).mul(k_sin));

        return .{ q_embed, k_embed };
    }

    pub fn rotateHalf(x: zml.Tensor) zml.Tensor {
        const half = @divExact(x.dim(.hd), 2);
        const x1 = x.slice1d(.hd, .{ .start = 0, .end = half });
        const x2 = x.slice1d(.hd, .{ .start = half, .end = x.dim(.hd) });

        return zml.Tensor.concatenate(&.{
            x2.negate(),
            x1,
        }, .hd);
    }

    pub fn repeatKv(hidden_states: zml.Tensor, n_rep: u32) zml.Tensor {
        if (n_rep == 1) return hidden_states;

        const with_group = hidden_states.insertAxes(.hd, .{.g}).broad(zml.Shape.init(.{
            .b = hidden_states.dim(.b),
            .s = hidden_states.dim(.s),
            .hk = hidden_states.dim(.hk),
            .g = n_rep,
            .hd = hidden_states.dim(.hd),
        }, hidden_states.dtype()));
        return with_group.merge(.{ .h = .{ .hk, .g } });
    }
};

pub const Qwen3MLP = struct {
    config: TextEncoder.Config,
    hidden_size: u32,
    intermediate_size: u32,
    gate_proj: zml.nn.Linear,
    up_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,
    act_fn: []const u8, // silu

    pub fn init(store: zml.io.TensorStore.View, config: TextEncoder.Config) Qwen3MLP {
        return .{
            .config = config,
            .hidden_size = config.hidden_size,
            .intermediate_size = config.intermediate_size,
            .up_proj = .init(
                store.withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .gate_proj = .init(
                store.withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, .{ .dout = .model, .d = .replicated }),
                null,
                .d,
            ),
            .down_proj = .init(
                store.withPrefix("down_proj").createTensor("weight", .{ .d, .dout }, .{ .d = .replicated, .dout = .model }),
                null,
                .dout,
            ),
            .act_fn = config.hidden_act,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Qwen3MLP)) void {
        self.up_proj.weight.deinit();
        if (self.up_proj.bias) |*bias| bias.deinit();
        self.gate_proj.weight.deinit();
        if (self.gate_proj.bias) |*bias| bias.deinit();
        self.down_proj.weight.deinit();
        if (self.down_proj.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: *const Qwen3MLP, x: zml.Tensor) zml.Tensor {
        const gate = self.gate_proj.forward(x);
        const up = self.up_proj.forward(x);

        const activated = if (std.mem.eql(u8, self.act_fn, "silu"))
            gate.silu()
        else
            @panic("unsupport4d");

        return self.down_proj.forward(activated.mul(up));
    }
};

test "repeatKv keeps repetitions adjacent to their source head" {
    const platform = zml.testing.env();
    const input: zml.Tensor = .init(.{ .b = 1, .s = 1, .hk = 2, .hd = 1 }, .f32);

    const forward = struct {
        fn forward(x: zml.Tensor) zml.Tensor {
            return Qwen3Attention.repeatKv(x, 2);
        }
    }.forward;

    var exe = try zml.module.compile(std.testing.allocator, std.testing.io, forward, .{input}, platform, .{});
    defer exe.deinit();

    var input_buffer = try zml.Buffer.fromBytes(
        std.testing.io,
        platform,
        input.shape(),
        .replicated,
        std.mem.sliceAsBytes(&[_]f32{ 10, 20 }),
    );
    defer input_buffer.deinit();

    var result = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, forward, .{input_buffer});
    defer result.deinit();

    try std.testing.expectEqual(
        [1][1][4][1]f32{.{.{ .{10}, .{10}, .{20}, .{20} }}},
        try result.getValue([1][1][4][1]f32, std.testing.io),
    );
}
