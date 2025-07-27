#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : track_propogate.py
@Date    : 2025/05/27
'''
import torch
from functools import lru_cache
import torch.nn.functional as F
from typing import List, Tuple, Optional

from rift.cbv.planning.pluto.utils.nuplan_state_utils import CarlaAgentState
from rift.util.torch_util import get_device_name


def compute_agents_vertices_torch(
    center:  torch.Tensor,  # [..., Tr, 2] – agent centres (x, y)
    heading: torch.Tensor,  # [..., Tr]    – heading in radians
    shape:   torch.Tensor   # [..., 2] or [..., Tr, 2] – [width, length]
) -> torch.Tensor:
    """
    Compute the four corner vertices of an oriented bounding box for each agent.

    Parameters
    ----------
    center : torch.Tensor
        Agent centre positions in world frame, shape [..., Tr, 2].
    heading : torch.Tensor
        Agent headings (yaw angles, radians), shape [..., Tr].
    shape : torch.Tensor
        Vehicle size given as [width, length].  
        It can be time‑invariant with shape [..., 2] or
        time‑variant with shape [..., Tr, 2].

    Returns
    -------
    torch.Tensor
        Corner vertices in world frame, shape [..., Tr, 4, 2],
        ordered as Front‑Left, Rear‑Left, Rear‑Right, Front‑Right.
    """

    # ──────────────────────────────
    # 1. Make shape time‑aligned
    # ──────────────────────────────
    if shape.dim() == center.dim() - 1:
        # Expand width/length over the time dimension without real copy
        shape = shape.unsqueeze(-2).expand(*center.shape[:-2], center.shape[-2], 2)

    half_w = 0.5 * shape[..., 0]          # [..., Tr]
    half_l = 0.5 * shape[..., 1]          # [..., Tr]

    # ──────────────────────────────
    # 2. Local corner offsets (vehicle frame)
    #    Shape [..., Tr, 4]
    # ──────────────────────────────
    dx = torch.stack(( half_l, -half_l, -half_l,  half_l), dim=-1)
    dy = torch.stack(( half_w,  half_w, -half_w, -half_w), dim=-1)

    # ──────────────────────────────
    # 3. Rotate offsets into world frame
    #    (avoid building full 2×2 matrices)
    # ──────────────────────────────
    cos_h = torch.cos(heading).unsqueeze(-1)  # [..., Tr, 1]
    sin_h = torch.sin(heading).unsqueeze(-1)

    x_world = dx * cos_h - dy * sin_h         # [..., Tr, 4]
    y_world = dx * sin_h + dy * cos_h         # [..., Tr, 4]

    # ──────────────────────────────
    # 4. Translate to world centres and stack
    # ──────────────────────────────
    vertices = torch.stack((x_world, y_world), dim=-1)     # [..., Tr, 4, 2]

    return vertices + center.unsqueeze(-2)                 # broadcast translation


def _heading_wrap(delta: torch.Tensor) -> torch.Tensor:
    """Wrap angle difference to (‑π, π]."""
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _central_diff(x: torch.Tensor, time_step: float) -> torch.Tensor:
    """
    Second‑order central difference with first‑order fallback at the ends.

    Parameters
    ----------
    x : torch.Tensor
        Shape (N, T, …) – N is any flattenable batch size,
        differentiation is along dim 1 (time).
    time_step : float
        Sampling period.

    Returns
    -------
    torch.Tensor
        Same shape as `x`, containing ∂x/∂t.
    """
    mid = (x[:, 2:]   - x[:, :-2]) / (2.0 * time_step)   # (N, T‑2, …)
    fst = (x[:, 1:2]  - x[:, :1])   /  time_step         # (N, 1 , …)
    lst = (x[:, -1:]  - x[:, -2:-1]) / time_step         # (N, 1 , …)
    return torch.cat([fst, mid, lst], dim=1)


def _sg_smooth(
    x: torch.Tensor,               # (N, 1, T)  – batched 1‑D signals
    window: int,                   # SG window length (must be odd)
    order: int,                    # polynomial order (< window)
    chunk_size: int = 30_000       # batch splitting to save memory
) -> torch.Tensor:
    """
    Savitzky–Golay smoothing (0‑th derivative) implemented with 1‑D convolution.

    * Works with FP16 and FP32 inputs.
    * The SG kernel is computed in higher precision when needed, then
      cast back to the original dtype for the convolution itself.
    """
    # ─── Sanity checks ──────────────────────────────────────────────────────────
    assert window % 2 == 1,   "`window` must be odd."
    assert order < window,    "`order` must be smaller than `window`."

    device      = x.device
    orig_dtype  = x.dtype                         # keep track of input precision

    # Use higher precision only to build the kernel if necessary (pinv not in FP16)
    calc_dtype  = torch.float32 if orig_dtype == torch.float16 else orig_dtype

    # ─── Kernel constructor with memoisation ───────────────────────────────────
    @lru_cache(maxsize=None)                      # reuse for identical args
    def _make_kernel(window: int,
                     order:  int,
                     dtype:  torch.dtype,
                     device: torch.device) -> torch.Tensor:
        """Return a 1‑D SG kernel (shape: [1, 1, window])."""
        half_len = window // 2
        t = torch.arange(-half_len,
                          half_len + 1,
                          dtype=dtype,
                          device=device)          # (window,)
        A = torch.stack([t ** i for i in range(order + 1)], dim=1)  # (window, order+1)
        pinv = torch.linalg.pinv(A.T @ A) @ A.T                      # (order+1, window)
        k0   = pinv[0].flip(0)                                      # 0‑th row → kernel
        return k0.view(1, 1, -1)                                    # (1, 1, window)

    # Build / fetch kernel, then cast to the original dtype
    kernel = _make_kernel(window, order, calc_dtype, device).to(orig_dtype)

    half_len = window // 2            # needed for padding below
    outputs  = []

    # ─── Chunked convolution to limit peak memory ──────────────────────────────
    for i in range(0, x.size(0), chunk_size):
        x_chunk = x[i:i + chunk_size]                                       # (b, 1, T)
        x_pad   = F.pad(x_chunk, (half_len, half_len), mode='reflect')      # (b, 1, T+2h)
        smoothed = F.conv1d(x_pad, kernel).squeeze(1)                       # (b, T)
        outputs.append(smoothed)

    return torch.cat(outputs, dim=0)    # (N, T)

class BatchKinematicBicycleModel:
    """
    Unified batch version of the kinematic bicycle model, handling both ego and surrounding vehicles.
    """

    def __init__(self, time_step=0.1):
        """
        Args:
            config: An object containing model parameters:
                - time_step
                - front_wheel_base
                - rear_wheel_base
                - steering_gain
                - brake_values (shape: [7,])
                - throttle_values (shape: [8,])
                - throttle_threshold_during_forecasting
        """
        device_name = get_device_name()
        self.device = torch.device(device_name)
        #  Time step for the model
        self.time_step = torch.tensor(time_step, dtype=torch.float32, device=self.device)

        # Kinematic bicycle model parameters tuned from World on Rails.
        # vehicle geometry (tuned from World on Rails)
        self.Lf = torch.tensor(-0.090769015, dtype=torch.float32, device=self.device)
        self.Lr = torch.tensor(1.4178275,    dtype=torch.float32, device=self.device)
        self.steering_gain = torch.tensor(0.36848336, dtype=torch.float32,
                                          device=self.device)

        # brake / throttle polynomial coefficients
        self.brake_values = torch.tensor(
            [9.31711370e-03, 8.20967431e-02, -2.83832427e-03, 5.06587474e-05,
             -4.90357228e-07, 2.44419284e-09, -4.91381935e-12],
            dtype=torch.float32, device=self.device
        )
        self.throttle_values = torch.tensor(
            [9.63873001e-01, 4.37535692e-04, -3.80192912e-01, 1.74950069e+00,
             9.16787414e-02, -7.05461530e-02, -1.05996152e-03, 6.71079346e-04],
            dtype=torch.float32, device=self.device
        )

        self.throttle_threshold = torch.tensor(0.3, dtype=torch.float32, device=self.device)

        # Kinematic bicycle model parameters tuned from World on Rails.
        # Distance from the rear axle to the front axle of the vehicle.
        self.front_wheel_base = torch.tensor(-0.090769015, dtype=torch.float32, device=self.device)
        # Distance from the rear axle to the center of the rear wheels.
        self.rear_wheel_base = torch.tensor(1.4178275, dtype=torch.float32, device=self.device)
        # Deceleration rate when braking (m/s^2) of other vehicles.
        self.brake_acceleration = torch.tensor(-4.952399, dtype=torch.float32, device=self.device)
        # Acceleration rate when throttling (m/s^2) of other vehicles.
        self.throttle_acceleration = torch.tensor(0.5633837, dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def forward(self,
                 locations: torch.Tensor,   # (G, 2)
                 headings:  torch.Tensor,   # (G,)
                 speeds:    torch.Tensor,   # (G,)   m/s
                 actions:   torch.Tensor    # (G, 3) throttle, steer, brake
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state for each vehicle in the batch.

        Returns
        -------
        next_locations : torch.Tensor (G, 2)
        next_headings  : torch.Tensor (G,)
        next_speeds    : torch.Tensor (G,)  non-negative, m/s
        """
        # ensure tensors are on the correct device / dtype
        locations = locations.to(self.device, dtype=torch.float32)
        headings  = headings.to(self.device,  dtype=torch.float32)
        speeds    = speeds.to(self.device,    dtype=torch.float32)
        actions   = actions.to(self.device,   dtype=torch.float32)

        throttle, steer, brake = actions.unbind(dim=1)
        brake_bool = brake.round().bool()        # treat >0.5 as braking

        # bicycle kinematics
        wheel_angle = self.steering_gain * steer
        slip_angle  = torch.atan((self.Lr / (self.Lf + self.Lr)) *
                                 torch.tan(wheel_angle))

        dx = speeds * torch.cos(headings + slip_angle) * self.time_step
        dy = speeds * torch.sin(headings + slip_angle) * self.time_step
        next_headings = headings + (speeds / self.Lr) * torch.sin(slip_angle) * self.time_step

        next_locations = locations.clone()
        next_locations[:, 0] += dx
        next_locations[:, 1] += dy

        # speed update in kph for polynomial convenience
        speed_kph = speeds * 3.6

        # brake polynomial: v¹ … v⁷
        v_powers = torch.stack([speed_kph.pow(i) for i in range(1, 8)], dim=1)
        next_speed_kph_brake = v_powers @ self.brake_values

        # throttle polynomial features
        v  = speed_kph
        v2 = v * v
        t  = throttle
        t2 = t * t
        throttle_feats = torch.stack(
            [v, v2, t, t2, v * t, v * t2, v2 * t, v2 * t2],
            dim=1
        )
        next_speed_kph_throttle = throttle_feats @ self.throttle_values

        # choose which update applies
        throttle_mask = (~brake_bool) & (throttle >= self.throttle_threshold)

        next_speed_kph = torch.where(brake_bool, next_speed_kph_brake, speed_kph)
        next_speed_kph = torch.where(throttle_mask, next_speed_kph_throttle,
                                     next_speed_kph)

        # convert back to m/s and clamp to non-negative
        next_speeds = torch.clamp(next_speed_kph / 3.6, min=0.0)

        return next_locations, next_headings, next_speeds
    
    @torch.inference_mode()
    def forecast_other_vehicles(self, locations: torch.Tensor, headings: torch.Tensor, speeds: torch.Tensor, actions: torch.Tensor):
        """
        Forecast the future states of other vehicles based on their current states and actions.
        
        Args:
            locations (torch.Tensor): Tensor of shape (N, 2) representing the (x, y) coordinates of other vehicles.
            headings (torch.Tensor): Tensor of shape (N,) representing the heading angles (in radians) for other vehicles.
            speeds (torch.Tensor): Tensor of shape (N,) representing the speeds (in m/s) for other vehicles.
            actions (torch.Tensor): Tensor of shape (N, 3) representing the actions (throttle, steer, brake) for other vehicles.
        
        Returns:
            tuple: A tuple containing the forecasted locations, headings, and speeds for other vehicles.
        """
        throttles, steers, brakes = actions[:, 0], actions[:, 1], actions[:, 2].to(torch.uint8)
        
        # Apply steering gain and calculate slip angle
        wheel_angles = self.steering_gain * steers
        slip_angles = torch.atan(self.rear_wheel_base / (self.front_wheel_base + self.rear_wheel_base) *
                                torch.tan(wheel_angles))

        # Forecast next positions
        next_x = locations[:, 0] + speeds * torch.cos(headings + slip_angles) * self.time_step
        next_y = locations[:, 1] + speeds * torch.sin(headings + slip_angles) * self.time_step
        next_headings = headings + speeds / self.rear_wheel_base * torch.sin(slip_angles) * self.time_step

        # Forecast next speeds with clamp to prevent negative values
        next_speeds = speeds + self.time_step * torch.where(brakes.bool(), self.brake_acceleration,
                                                            throttles * self.throttle_acceleration)
        next_speeds = torch.maximum(torch.tensor(0.0, device=locations.device), next_speeds)

        # Combine forecasted positions
        next_locations = torch.stack([next_x, next_y], dim=1)

        return next_locations, next_headings, next_speeds


class BatchPIDTorch:
    """GPU-friendly batched PID controller."""

    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20,
                 device = None,
                 dtype = torch.float32):
        self.K_P, self.K_I, self.K_D = K_P, K_I, K_D
        self.n = n
        self.device = torch.device(device) if device else torch.device("cpu")
        self.dtype = dtype

        # Lazily initialised buffers
        self._buf = None          # (B, n)
        self._ptr = None          # (B,)
        self._len = None          # (B,)
        self._max_error = None    # (B,)

    # ------------------------------------------------------------------ #
    # Allocate / extend per-trajectory state to match current batch size
    # ------------------------------------------------------------------ #
    def _ensure_capacity(self, B: int):
        if self._buf is None:                 # first call
            self._buf       = torch.zeros((B, self.n), dtype=self.dtype, device=self.device)
            self._ptr       = torch.zeros(B, dtype=torch.long, device=self.device)
            self._len       = torch.zeros(B, dtype=torch.long, device=self.device)
            self._max_error = torch.zeros(B, dtype=self.dtype, device=self.device)
        elif self._buf.size(0) < B:           # batch has grown → pad arrays
            extra = B - self._buf.size(0)
            self._buf       = torch.cat([self._buf,
                                         torch.zeros((extra, self.n), dtype=self.dtype, device=self.device)], dim=0)
            self._ptr       = torch.cat([self._ptr,
                                         torch.zeros(extra, dtype=torch.long, device=self.device)], dim=0)
            self._len       = torch.cat([self._len,
                                         torch.zeros(extra, dtype=torch.long, device=self.device)], dim=0)
            self._max_error = torch.cat([self._max_error,
                                         torch.zeros(extra, dtype=self.dtype, device=self.device)], dim=0)

    # ------------------------------------------------------------------ #
    # Optional: clear integral / derivative history
    # ------------------------------------------------------------------ #
    def reset(self):
        if self._buf is not None:
            self._buf.zero_()
            self._ptr.zero_()
            self._len.zero_()
            self._max_error.zero_()

    # ------------------------------------------------------------------ #
    # Vectorised PID update
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def step(self, error: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        error : (B,) torch.Tensor
            Current error for each trajectory (should already be on self.device).

        Returns
        -------
        (B,) torch.Tensor : PID output
        """
        error = error.to(device=self.device, dtype=self.dtype)
        B = error.size(0)
        self._ensure_capacity(B)

        idx = self._ptr[:B]                                    # where to write
        prev_error = self._buf[torch.arange(B, device=self.device), idx]

        # Circular buffer update
        self._buf[torch.arange(B, device=self.device), idx] = error
        self._ptr[:B] = (idx + 1) % self.n
        self._len[:B] = torch.clamp(self._len[:B] + 1, max=self.n)

        # Track running |error| max (kept only to mirror your original code)
        self._max_error[:B] = torch.maximum(self._max_error[:B], error.abs())

        integral   = self._buf[:B].sum(dim=1) / self._len[:B].clamp(min=1).to(self.dtype)
        derivative = error - prev_error

        return (self.K_P * error +
                self.K_I * integral +
                self.K_D * derivative)


class BatchPIDController:
    """Batched version of your PIDController that runs entirely on torch."""

    def __init__(self, sample_interval=10, max_throttle=0.75,
                 brake_speed=0.4, brake_ratio=1.1, clip_delta=0.25,
                 dtype: torch.dtype = torch.float32):
        device_name = get_device_name()
        self.device = torch.device(device_name)
        self.dtype = dtype

        self.sample_interval = int(sample_interval)

        self.turn_controller  = BatchPIDTorch(1.25, 0.75, 0.3, n=20,
                                              device=self.device, dtype=self.dtype)
        self.speed_controller = BatchPIDTorch(5.0,  0.5,  1.0, n=20,
                                              device=self.device, dtype=self.dtype)

        # Hyper-parameters
        self.alpha, self.beta = 0.5, 2.5
        self.min_aim_dis, self.max_aim_dis = 5.0, 8.0
        self.max_throttle = max_throttle
        self.brake_speed  = brake_speed
        self.brake_ratio  = brake_ratio
        self.clip_delta   = clip_delta

        # Debug hooks
        self.desired_speed = None
        self.delta_angle   = None

    # ------------------------------------------------------------------ #
    # Main control step (all GPU)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def control_pid(self,
                    local_pos: torch.Tensor,   # (B, T, 2)
                    speed:     torch.Tensor):  # (B,)
        """Return action tensor (B, 3) throttle, steer, brake on same device as controller."""

        local_pos = local_pos.to(device=self.device, dtype=self.dtype)
        speed     = speed.to(device=self.device, dtype=self.dtype)
        B, T, _ = local_pos.shape

        # 1. Resample waypoints ----------------------------------------------------
        if T >= self.sample_interval:
            local_pos_rs = local_pos[:, self.sample_interval-1::self.sample_interval]
        else:                                   # take the very last waypoint
            local_pos_rs = local_pos[:, -1:, :]

        # 2. Desired speed ---------------------------------------------------------
        seg_vec = local_pos_rs[:, 1:] - local_pos_rs[:, :-1]      # (B, T'-1, 2)
        if seg_vec.numel() == 0:                                 # only one waypoint
            desired_v = torch.zeros(B, device=self.device, dtype=self.dtype)
        else:
            seg_len = seg_vec.norm(dim=2)                        # (B, T'-1)
            desired_v = seg_len.mean(dim=1)                      # (B,)

        # 3. Aim point -------------------------------------------------------------
        aim_dist = torch.clamp(self.alpha * speed + self.beta,
                               min=self.min_aim_dis,
                               max=self.max_aim_dis)              # (B,)

        if local_pos_rs.size(1) == 1:
            aim = local_pos_rs[:, 0]                             # (B, 2)
        else:
            norms = local_pos_rs[:, :-1].norm(dim=2)             # (B, T'-1)
            idx = (norms - aim_dist[:, None]).abs().argmin(dim=1)  # (B,)
            aim = local_pos_rs[torch.arange(B, device=self.device), idx]  # (B, 2)

        # 4. Brake / throttle ------------------------------------------------------
        brake = (desired_v < self.brake_speed) | \
                ((speed / desired_v.clamp(min=1e-4)) > self.brake_ratio)

        delta    = torch.clamp(desired_v - speed, min=0.0, max=self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = torch.clamp(throttle, min=0.0, max=self.max_throttle)
        throttle = throttle * (~brake)                           # zero throttle if braking

        # 5. Steering --------------------------------------------------------------
        angle = torch.rad2deg(torch.atan2(aim[:, 1], aim[:, 0])) / 90.0  # IMPORTANT !! for right-hand execution
        angle = torch.where((speed < 0.01) | brake, torch.zeros_like(angle), angle)

        steer = self.turn_controller.step(angle)
        steer = torch.clamp(steer, min=-1.0, max=1.0)

        # 6. Debug -----------------------------------------------------------------
        self.desired_speed = desired_v
        self.delta_angle   = angle

        return torch.stack([throttle, steer, brake], dim=1)  # (B, 3)

    # ------------------------------------------------------------------ #
    # Reset both embedded PID filters
    # ------------------------------------------------------------------ #
    def reset(self):
        self.turn_controller.reset()
        self.speed_controller.reset()

def derive_kinematics(
    headings: torch.Tensor,                       # [..., T]
    positions: Optional[torch.Tensor] = None,     # [..., T, 2]  (exclusive)
    speed:     Optional[torch.Tensor] = None,     # [..., T]
    time_step: float = 0.1,
    smooth_window: Optional[int] = 5,             # odd ⇒ enable SG smoothing
    smooth_order:  int = 2
) -> Tuple[torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """
    Estimate physically consistent speed, acceleration, yaw‑rate,
    and yaw‑acceleration from either positions *or* speed plus headings.

    Works with any leading dimensions; results are returned in the
    same layout.

    Returns
    -------
    speed              : [..., T]
    acceleration       : [..., T]
    yaw_rate           : [..., T]
    yaw_acceleration   : [..., T]
    """
    # ------------------------------------------------------------
    # 0)  Input checks and flatten to (N_flat, T)
    # ------------------------------------------------------------
    assert (positions is not None) or (speed is not None), \
    "Provide at least one of `positions` or `speed`."

    lead_shape = headings.shape[:-1]               # arbitrary leading dims
    T = headings.shape[-1]
    N_flat = int(torch.prod(torch.tensor(lead_shape))) or 1

    device, dtype = headings.device, headings.dtype
    head_f = headings.reshape(N_flat, T)

    # Short sequences → zeros
    if T < 3:
        zeros = torch.zeros_like(head_f)
        zeros_full = zeros.reshape(*lead_shape, T)
        return zeros_full, zeros_full, zeros_full, zeros_full

    # ------------------------------------------------------------
    # 1)  Linear speed
    # ------------------------------------------------------------
    if speed is not None:
        speed_f = speed.reshape(N_flat, T).to(device=device, dtype=dtype)
    else:
        # Positions given, compute speed from them
        pos_f = positions.reshape(N_flat, T, 2)

        disp_mid  = pos_f[:, 2:] - pos_f[:, :-2]               # (N, T‑2, 2)
        speed_mid = disp_mid.norm(dim=-1) / (2 * time_step)    # (N, T‑2)

        speed_f = torch.zeros_like(head_f)
        speed_f[:, 1:-1] = speed_mid
        speed_f[:, 0]  = (pos_f[:, 1]  - pos_f[:, 0]).norm(dim=-1) / time_step
        speed_f[:, -1] = (pos_f[:, -1] - pos_f[:, -2]).norm(dim=-1) / time_step

    # Optional Savitzky–Golay smoothing (0‑th order only)
    if smooth_window is not None and smooth_window > 2 and smooth_window % 2 == 1:
        speed_f = _sg_smooth(speed_f.unsqueeze(1), smooth_window, smooth_order)

    # ------------------------------------------------------------
    # 2)  Linear acceleration
    # ------------------------------------------------------------
    accel_f = _central_diff(speed_f.unsqueeze(-1), time_step).squeeze(-1)

    # ------------------------------------------------------------
    # 3)  Yaw‑rate
    # ------------------------------------------------------------
    head_proc = head_f
    if smooth_window is not None and smooth_window > 2 and smooth_window % 2 == 1:
        head_proc = _sg_smooth(head_f.unsqueeze(1), smooth_window, smooth_order)

    dtheta_mid = _heading_wrap(head_proc[:, 2:] - head_proc[:, :-2])

    yaw_rate_f = torch.zeros_like(head_f)
    yaw_rate_f[:, 1:-1] = dtheta_mid / (2 * time_step)
    yaw_rate_f[:, 0]  = _heading_wrap(head_proc[:, 1]  - head_proc[:, 0])  / time_step
    yaw_rate_f[:, -1] = _heading_wrap(head_proc[:, -1] - head_proc[:, -2]) / time_step

    # ------------------------------------------------------------
    # 4)  Yaw‑acceleration
    # ------------------------------------------------------------
    yaw_accel_f = _central_diff(yaw_rate_f.unsqueeze(-1), time_step).squeeze(-1)

    # ------------------------------------------------------------
    # 5)  Restore original leading dims
    # ------------------------------------------------------------
    out_shape = (*lead_shape, T)
    speed_out          = speed_f.reshape(out_shape)
    accel_out          = accel_f.reshape(out_shape)
    yaw_rate_out       = yaw_rate_f.reshape(out_shape)
    yaw_accel_out      = yaw_accel_f.reshape(out_shape)

    return speed_out, accel_out, yaw_rate_out, yaw_accel_out


class TrackPropagate:
    """
    Class to handle the propagation of tracks in a trajectory evaluation context.
    This class is a placeholder for future implementations and currently does not contain any methods or attributes.
    """
    def __init__(self, virtual_time_step=0.1, rollout_length=80):
        """
        Initializes the TrackPropagate class with a virtual time step and initializes the PID controller and kinematic bicycle model.
        Args:
            virtual_time_step (float): The time step for the propagation, default is 0.1 seconds.
        """
        self.virtual_time_step = virtual_time_step
        self.rollout_length = rollout_length
        self.device = get_device_name()
        self.batch_pid_controller = BatchPIDController()
        self.batch_kinematic_bicycle_model = BatchKinematicBicycleModel(self.virtual_time_step)

    def propagate(self,
        ref_traj_pos: torch.Tensor, # [G, T, 2]
        ref_traj_heading: torch.Tensor,  # [G, T]
        center_history_states: List[CarlaAgentState]
    ):
        """
        Placeholder method for propagating tracks.
        To be implemented in the future.
        """
        ego_state = center_history_states[-1]
        G, T, C = ref_traj_pos.shape
        device, dtype = ref_traj_pos.device, ref_traj_pos.dtype
        center_cur_speed = ego_state.dynamic_car_state.speed

        # Pre-allocate arrays for storing results
        rollout_center = [ref_traj_pos[:, 0, :]]                                   # List[[G, 2]]
        rollout_angle = [ref_traj_heading[:, 0]]                                 # List[[G]]
        rollout_speed = [torch.full((G,), center_cur_speed, device=device, dtype=dtype)]             # List[[G]]

        forward_step  = 0
        closest_index = torch.zeros(G, device=device, dtype=torch.long)  # [G,]

        while forward_step < self.rollout_length - 1:  # while the last point is not reached
            # get the current local trajectory pos for PID control
            local_traj_pos = self.get_local_traj_pos(
                ref_traj_pos, closest_index, 
                rollout_center[-1], rollout_angle[-1]
            )  # [G, T, 2]
            
            # compute the PID control actions (throttle, steer, brake)
            actions = self.batch_pid_controller.control_pid(
                local_pos=local_traj_pos, 
                speed=rollout_speed[-1]
            )  # [G, 3] throttle, steer, brake

            # compute the kinematic model
            next_pos, next_headings, next_speeds = self.batch_kinematic_bicycle_model.forward(
                locations=rollout_center[-1],            # [G, 2]
                headings=rollout_angle[-1],              # [G,]
                speeds=rollout_speed[-1],                # [G,]
                actions=actions                          # [G, 3] throttle, steer, brake
            )

            rollout_center.append(next_pos)              # [G, 2]
            rollout_angle.append(next_headings)          # [G,]
            rollout_speed.append(next_speeds)            # [G,]
            
            # --- update closest index & stall counter --------------------------------
            closest_index = self.find_closest_ref_pos(ref_traj_pos, next_pos)

            forward_step += 1
        
        # -------------------------- final rollout data (in float16) -------------------------- #
        rollout_center = torch.stack(rollout_center, dim=1)  # [G, Tr, 2]
        rollout_angle = torch.stack(rollout_angle, dim=1)   # [G, Tr]
        rollout_speed = torch.stack(rollout_speed, dim=1)       # [G, Tr]

        # approximate the acceleration through forward finite differences.
        rollout_speed, rollout_acc, rollout_angular_vel, rollout_angular_acc = derive_kinematics(
            headings=rollout_angle, speed=rollout_speed, time_step=self.virtual_time_step
        )

        # vehicle shape
        center_shape = torch.tensor(
            [ego_state.car_footprint.width, ego_state.car_footprint.length],
            dtype=dtype,
            device=device
        ).expand(G, -1)  # (G, 2)

        vertices = compute_agents_vertices_torch(
            center=rollout_center,    # [G, Tr, 2]
            heading=rollout_angle,  # [G, Tr]
            shape=center_shape,       # [G, 2]
        )  # global coord, right-hand   [G, Tr, 4, 2]

        rollout_center = rollout_center.cpu().numpy()
        rollout_angle = rollout_angle.cpu().numpy()
        rollout_speed = rollout_speed.cpu().numpy()
        rollout_acc = rollout_acc.cpu().numpy() 
        rollout_angular_vel = rollout_angular_vel.cpu().numpy()
        rollout_angular_acc = rollout_angular_acc.cpu().numpy()
        vertices = vertices.cpu().numpy()  # [G, Tr, 4, 2]

        return rollout_center, rollout_angle, rollout_speed, rollout_acc, rollout_angular_vel, rollout_angular_acc, vertices

    def get_local_traj_pos(self,
        ref_traj_pos: torch.Tensor,
        closest_index: torch.Tensor,
        current_pos: torch.Tensor,
        current_heading: torch.Tensor = None,
        future_len: int = 30
    ) -> torch.Tensor:
        """
        Extract future_len points from closest_index onward for each trajectory,
        padding with the trajectory's last point if not enough points remain.
        Then shift all points to be relative to closest_ref_pos.

        Args:
            ref_traj_pos: (G, T, 2), global trajectory
            closest_index: (G,), index of closest point on each trajectory
            current_pos: (G, 2), current position of the vehicle
            current_heading: (G,), current heading of the vehicle
            future_len: int, number of points to extract

        Returns:
            local_traj_pos: (G, future_len, 2), relative trajectory in local frame
        """
        G, T, _ = ref_traj_pos.shape
        device = ref_traj_pos.device

        # Create index offsets
        idx_offsets = torch.arange(future_len, device=device).unsqueeze(0)  # (1, future_len)
        idx = closest_index.unsqueeze(1) + idx_offsets  # (G, future_len)
        idx_clipped = idx.clamp(max=T - 1)  # (G, future_len)

        # Gather valid points
        batch_idx = torch.arange(G, device=device).unsqueeze(1).expand(-1, future_len)  # (G, future_len)
        traj_gathered = ref_traj_pos[batch_idx, idx_clipped]  # (G, future_len, 2)

        # Mask for padding (True if index exceeds bounds)
        padding_mask = idx >= T  # (G, future_len)
        padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, 2)  # (G, future_len, 2)

        # Get last point of each trajectory
        last_points = ref_traj_pos[:, -1:, :]  # (G, 1, 2)
        last_points = last_points.expand(-1, future_len, -1)  # (G, future_len, 2)

        # Replace out-of-bound positions with last point
        traj_padded = torch.where(padding_mask, last_points, traj_gathered)  # (G, future_len, 2)

        # Translate the trajectory to be relative to the current position
        local_traj_pos = traj_padded - current_pos.unsqueeze(1)  # (G, future_len, 2)
        # Rotate to the local frame using heading
        cos_h = torch.cos(current_heading)
        sin_h = torch.sin(current_heading)
        rot = torch.stack(
            (
                torch.stack((cos_h, -sin_h), dim=-1),
                torch.stack((sin_h,  cos_h), dim=-1),
            ),
            dim=-2
        )
        local_traj_pos = torch.matmul(local_traj_pos, rot)  # (G, future_len, 2)

        return local_traj_pos

    def find_closest_ref_pos(self, ref_traj_pos: torch.Tensor, next_pos: torch.Tensor):
        """
        For each of the G input positions in next_pos, find the closest point on the corresponding trajectory in ref_traj_pos.

        Args:
            ref_traj_pos: torch.Tensor of shape (G, T, 2), reference trajectories
            next_pos: torch.Tensor of shape (G, 2), query points

        Returns:
            closest_index: torch.Tensor of shape (G,), the index of the closest point along the trajectory
        """
        # Compute the squared distance between next_pos and each point in ref_traj_pos
        diff = ref_traj_pos - next_pos[:, None, :]     # (G, T, 2)
        dist_sq = (diff ** 2).sum(dim=-1)              # (G, T)

        # Find the index of the closest point
        closest_index = torch.argmin(dist_sq, dim=1)   # (G,)

        return closest_index