const std = @import("std");

const zml = @import("zml");

// The configs deffined here depend on the models scheduler/scheduler_config.json
// One can find it here := https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/scheduler/scheduler_config.json
pub const SchedulerConfig = struct {
    num_train_timesteps: u32 = 1000,
    use_dynamic_shifting: bool = false,
    shift: f32 = 6.0,
};

pub const TimeShift = enum { exponential, linear };

pub const SchedulerOutput = struct {
    prev_sample: zml.Tensor,
};

pub const StepSigmas = struct {
    current: f32,
    next: f32,
};

// ZImagePipeline uses FlowMatchEulerDiscreteScheduler
pub const Scheduler = struct {
    num_train_timesteps: u32 = 1000,
    shift: f32 = 1.0,
    use_dynamic_shifting: bool = false,
    base_shift: ?f32 = 0.5,
    max_shift: ?f32 = 1.15,
    base_image_seq_len: u32 = 256,
    max_image_seq_len: u32 = 4096,
    invert_sigmas: bool = false,
    shift_terminal: ?f32 = null,
    use_karras_sigmas: bool = false,
    use_exponential_sigmas: bool = false,
    use_beta_sigmas: bool = false,
    time_shift_type: TimeShift = TimeShift.exponential,
    stochastic_sampling: bool = false,
    timesteps: []f32,
    sigmas: []f32,
    sigma_min: f32,
    sigma_max: f32,
    step_index: ?u32 = null,
    begin_index: ?u32 = null,

    pub fn init(allocator: std.mem.Allocator, config: SchedulerConfig) !Scheduler {
        if (@intFromBool(false) + @intFromBool(false) + @intFromBool(false) > 1) {
            return error.InvalidSigmaConfiguration;
        }

        const timesteps = try allocator.alloc(f32, config.num_train_timesteps);
        errdefer allocator.free(timesteps);
        const sigmas = try allocator.alloc(f32, config.num_train_timesteps);
        errdefer allocator.free(sigmas);

        const num_train_timesteps_f32 = @as(f32, @floatFromInt(config.num_train_timesteps));
        for (0..config.num_train_timesteps) |i| {
            const t = @as(f32, @floatFromInt(config.num_train_timesteps - i));
            var sigma = t / num_train_timesteps_f32;
            if (!config.use_dynamic_shifting) {
                sigma = config.shift * sigma / (1 + (config.shift - 1) * sigma);
            }

            sigmas[i] = sigma;
            timesteps[i] = sigma * num_train_timesteps_f32;
        }

        return .{
            .num_train_timesteps = config.num_train_timesteps,
            .shift = config.shift,
            .use_dynamic_shifting = config.use_dynamic_shifting,
            .base_shift = 0.5,
            .max_shift = 1.15,
            .base_image_seq_len = 256,
            .max_image_seq_len = 4096,
            .invert_sigmas = false,
            .shift_terminal = null,
            .use_karras_sigmas = false,
            .use_exponential_sigmas = false,
            .use_beta_sigmas = false,
            .time_shift_type = .exponential,
            .stochastic_sampling = false,
            .timesteps = timesteps,
            .sigmas = sigmas,
            .sigma_min = sigmas[sigmas.len - 1],
            .sigma_max = sigmas[0],
            .step_index = null,
            .begin_index = null,
        };
    }

    pub fn deinit(self: Scheduler, allocator: std.mem.Allocator) void {
        allocator.free(self.timesteps);
        allocator.free(self.sigmas);
    }

    pub fn unbuffered(self: Scheduler) Scheduler {
        return self;
    }

    fn initStepIndex(self: *Scheduler, timestep: f32) void {
        if (self.step_index != null) return;

        var best_idx: u32 = 0;
        var best_dist = std.math.floatMax(f32);
        for (self.timesteps, 0..) |t, i| {
            const dist = @abs(t - timestep);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = @intCast(i);
            }
        }
        self.step_index = best_idx;
    }

    pub fn nextStepSigmas(self: *Scheduler, timestep: f32) StepSigmas {
        if (self.step_index == null) {
            self.initStepIndex(timestep);
        }

        const sigma_idx = self.step_index.?;
        const current_sigma = self.sigmas[sigma_idx];
        const next_sigma = self.sigmas[@min(sigma_idx + 1, self.sigmas.len - 1)];

        self.step_index = sigma_idx + 1;

        return .{
            .current = current_sigma,
            .next = next_sigma,
        };
    }

    pub fn setTimesteps(self: *Scheduler, allocator: std.mem.Allocator, num_inference_steps: u32, mu: ?f32) !void {
        if (self.use_dynamic_shifting and mu == null) return error.MissingDynamicShift;
        if (num_inference_steps == 0) return error.InvalidNumInferenceSteps;

        const timesteps = try allocator.alloc(f32, num_inference_steps);
        errdefer allocator.free(timesteps);
        const sigmas = try allocator.alloc(f32, num_inference_steps + 1);
        errdefer allocator.free(sigmas);

        const num_train_timesteps_f32 = @as(f32, @floatFromInt(self.num_train_timesteps));
        const num_inference_steps_f32 = @as(f32, @floatFromInt(num_inference_steps));

        for (0..num_inference_steps) |i| {
            // ZImagePipeline supplies linspace(1, 1 / steps, steps) before the
            // FlowMatch scheduler applies its configured shift.
            const alpha = @as(f32, @floatFromInt(i)) / num_inference_steps_f32;
            var sigma = 1.0 - alpha;

            if (self.use_dynamic_shifting) {
                const mu_ = mu.?;
                sigma = switch (self.time_shift_type) {
                    .exponential => blk: {
                        const exp_mu = std.math.exp(mu_);
                        break :blk exp_mu / (exp_mu + std.math.pow(f32, 1.0 / sigma - 1.0, 1.0));
                    },
                    .linear => mu_ / (mu_ + 1.0 / sigma - 1.0),
                };
            } else {
                sigma = self.shift * sigma / (1 + (self.shift - 1) * sigma);
            }

            sigmas[i] = sigma;
            timesteps[i] = sigma * num_train_timesteps_f32;
        }

        sigmas[num_inference_steps] = 0.0;

        allocator.free(self.timesteps);
        allocator.free(self.sigmas);
        self.timesteps = timesteps;
        self.sigmas = sigmas;
        self.sigma_min = sigmas[num_inference_steps - 1];
        self.sigma_max = sigmas[0];
        self.step_index = null;
        self.begin_index = null;
    }

    pub fn step(
        self: *Scheduler,
        model_output: zml.Tensor,
        timestep: f32,
        sample: zml.Tensor,
        per_token_timesteps: ?zml.Tensor,
    ) SchedulerOutput {
        if (self.step_index == null) {
            self.initStepIndex(timestep);
        }

        const sample_f32 = sample.convert(.f32);

        const prev_sample = if (per_token_timesteps) |token_timesteps| blk: {
            const per_token_sigmas = token_timesteps.convert(.f32).div(zml.Tensor.scalar(@as(f32, @floatFromInt(self.num_train_timesteps)), .f32));

            var lower_sigma: f32 = self.sigmas[self.sigmas.len - 1];
            for (self.sigmas) |sigma| {
                if (sigma < timestep / @as(f32, @floatFromInt(self.num_train_timesteps)) - 1e-6) {
                    lower_sigma = sigma;
                    break;
                }
            }

            const current_sigma = per_token_sigmas.insertAxes(.last, .{.sigma});
            const next_sigma = zml.Tensor.scalar(lower_sigma, .f32).broad(current_sigma.shape());
            const dt = current_sigma.sub(next_sigma);

            if (self.stochastic_sampling) {
                const x0 = sample_f32.sub(current_sigma.mul(model_output.convert(.f32)));
                const noise = zml.Tensor.zeros(sample_f32.shape(), .f32);
                break :blk zml.Tensor.scalar(1.0, .f32)
                    .sub(next_sigma)
                    .mul(x0)
                    .add(next_sigma.mul(noise));
            }

            break :blk sample_f32.add(dt.mul(model_output.convert(.f32)));
        } else blk: {
            const sigma_idx = self.step_index.?;
            const sigma = self.sigmas[sigma_idx];
            const sigma_next = self.sigmas[@min(sigma_idx + 1, self.sigmas.len - 1)];
            const current_sigma = sigma;
            const next_sigma = sigma_next;
            const dt = next_sigma - current_sigma;

            if (self.stochastic_sampling) {
                const x0 = sample_f32.sub(model_output.convert(.f32).scale(current_sigma));
                const noise = zml.Tensor.zeros(sample_f32.shape(), .f32);
                break :blk x0.scale(1.0 - next_sigma).add(noise.scale(next_sigma));
            }

            break :blk sample_f32.add(model_output.convert(.f32).scale(dt)).convert(model_output.dtype());
        };

        self.step_index = (self.step_index orelse 0) + 1;

        return .{ .prev_sample = prev_sample };
    }
};

test "Z-Image sigma schedule matches FlowMatch defaults" {
    const allocator = std.testing.allocator;
    var scheduler = try Scheduler.init(allocator, .{});
    defer scheduler.deinit(allocator);

    try scheduler.setTimesteps(allocator, 4, null);

    const expected_unshifted = [_]f32{ 1.0, 0.75, 0.5, 0.25 };
    for (expected_unshifted, scheduler.sigmas[0..4], scheduler.timesteps) |base, sigma, timestep| {
        const expected = 6.0 * base / (1.0 + 5.0 * base);
        try std.testing.expectApproxEqAbs(expected, sigma, 1e-6);
        try std.testing.expectApproxEqAbs(expected * 1000.0, timestep, 1e-3);
    }
    try std.testing.expectEqual(@as(f32, 0.0), scheduler.sigmas[4]);
}
