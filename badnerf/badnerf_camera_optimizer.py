"""
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from typing import Literal, Optional, Type, Union

import pypose as pp
import torch
from dataclasses import dataclass, field
from jaxtyping import Float, Int
from pypose import LieTensor
from torch import Tensor
from typing_extensions import assert_never

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig

from badnerf.spline_functor import (
    cubic_bspline_interpolation,
    linear_interpolation,
    linear_interpolation_mid,
)


TrajSamplingMode = Literal["uniform", "start", "mid", "end"]
"""How to sample the camera trajectory of blur images"""


@dataclass
class BadNerfCameraOptimizerConfig(CameraOptimizerConfig):
    """Configuration of BAD-NeRF camera optimizer."""

    _target: Type = field(default_factory=lambda: BadNerfCameraOptimizer)
    """The target class to be instantiated."""

    mode: Literal["off", "linear", "cubic"] = "off"
    """Pose optimization strategy to use.
    linear: linear interpolation on SE(3);
    cubic: cubic b-spline interpolation on SE(3)."""

    trans_l2_penalty: float = 0.0
    """L2 penalty on translation parameters."""

    rot_l2_penalty: float = 0.0
    """L2 penalty on rotation parameters."""

    num_virtual_views: int = 10
    """The number of samples used to model the motion-blurring."""

    initial_noise_se3_std: float = 1e-5
    """Initial perturbation to pose delta on se(3). Must be non-zero to prevent NaNs."""


class BadNerfCameraOptimizer(CameraOptimizer):
    """Optimization for BAD-NeRF virtual camera trajectories."""
    config: BadNerfCameraOptimizerConfig

    def __init__(
        self,
        config: BadNerfCameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__(CameraOptimizerConfig(), num_cameras, device)
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices
        self.dof = 6
        """Degrees of freedom of manifold, i.e. number of dimensions of the tangent space"""
        self.dim = 7
        """Dimentions of pose parameterization. Three for translation, 4-tuple for quaternion"""

        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "linear":
            self.num_control_knots = 2
        elif self.config.mode == "cubic":
            self.num_control_knots = 4
        else:
            assert_never(self.config.mode)

        self.pose_adjustment = pp.Parameter(
            pp.randn_se3(
                (num_cameras, self.num_control_knots),
                sigma=self.config.initial_noise_se3_std,
                device=device,
            ),
        )

    def forward(
        self,
        indices: Int[Tensor, "camera_indices"],
        mode: TrajSamplingMode = "mid",
    ) -> Float[LieTensor, "camera_indices self.num_control_knots self.dof"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
            mode: interpolate between start and end, or return start / mid / end.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        else:
            indices = indices.int()
            unique_indices, lut = torch.unique(indices, return_inverse=True)
            camera_opt = self.pose_adjustment[unique_indices].Exp()
            outputs.append(self._interpolate(camera_opt, mode)[lut])

        # Detach non-trainable indices by setting to identity transform
        if (
                torch.is_grad_enabled()
                and self.non_trainable_camera_indices is not None
                and len(indices) > len(self.non_trainable_camera_indices)
        ):
            if self.non_trainable_camera_indices.device != self.pose_adjustment.device:
                self.non_trainable_camera_indices = self.non_trainable_camera_indices.to(self.pose_adjustment.device)
            nt = self.non_trainable_camera_indices
            outputs[0][nt] = outputs[0][nt].clone().detach()

        # Return: identity if no transforms are needed, otherwise composite transforms together.
        if len(outputs) == 0:
            return pp.identity_SE3(
                *(indices.shape[0], self.num_control_knots),
                device=self.pose_adjustment.device
            )
        return functools.reduce(pp.mul, outputs)

    def _interpolate(
            self,
            camera_opt: Float[LieTensor, "*batch_size self.num_control_knots self.dof"],
            mode: TrajSamplingMode
    ) -> Float[Tensor, "*batch_size interpolations self.dof"]:
        if mode == "uniform":
            u = torch.linspace(
                start=0,
                end=1,
                steps=self.config.num_virtual_views,
                device=camera_opt.device,
            )
            if self.config.mode == "linear":
                return linear_interpolation(camera_opt, u)
            elif self.config.mode == "cubic":
                return cubic_bspline_interpolation(camera_opt, u)
            else:
                assert_never(self.config.mode)
        elif mode == "mid":
            if self.config.mode == "linear":
                return linear_interpolation_mid(camera_opt)
            elif self.config.mode == "cubic":
                return cubic_bspline_interpolation(
                    camera_opt,
                    torch.tensor([0.5], device=camera_opt.device)
                ).squeeze()
            else:
                assert_never(self.config.mode)
        elif mode == "start":
            if self.config.mode == "linear":
                return camera_opt[..., 0, :]
            elif self.config.mode == "cubic":
                return cubic_bspline_interpolation(
                    camera_opt,
                    torch.tensor([0.0], device=camera_opt.device)
                ).squeeze()
            else:
                assert_never(self.config.mode)
        elif mode == "end":
            if self.config.mode == "linear":
                return camera_opt[..., 1, :]
            elif self.config.mode == "cubic":
                return cubic_bspline_interpolation(
                    camera_opt,
                    torch.tensor([1.0], device=camera_opt.device)
                ).squeeze()
            else:
                assert_never(self.config.mode)
        else:
            assert_never(mode)

    def apply_to_raybundle(
            self,
            ray_bundle: RayBundle,
            mode: TrajSamplingMode,
    ) -> RayBundle:
        """Apply the pose correction to the raybundle"""
        assert ray_bundle.camera_indices is not None
        assert self.pose_adjustment.device == ray_bundle.origins.device

        if self.config.num_virtual_views == 1:
            return ray_bundle

        # duplicate optimized_bundle num_virtual_views times and stack
        def repeat_fn(x):
            return x.repeat_interleave(self.config.num_virtual_views, dim=0)

        camera_ids = ray_bundle.camera_indices.squeeze()
        if camera_ids.dim() == 0:
            camera_ids = camera_ids[None]

        poses_delta = self(camera_ids, mode)

        if mode == "uniform":
            if ray_bundle.ndim == 2:
                ray_bundle = ray_bundle._apply_fn_to_fields(lambda x: x.flatten(start_dim=0, end_dim=1))
                poses_delta = pp.SE3(torch.flatten(poses_delta, start_dim=0, end_dim=1))
            poses_delta = pp.SE3(torch.flatten(poses_delta, start_dim=0, end_dim=1))
            ray_bundle = ray_bundle._apply_fn_to_fields(repeat_fn)

        t = poses_delta.translation()
        q = poses_delta.rotation()
        ray_bundle.origins = ray_bundle.origins + t
        ray_bundle.directions = pp.SO3(q) @ ray_bundle.directions

        return ray_bundle

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        if self.config.mode != "off":
            metrics_dict["camera_opt_trajectory_translation"] = (
                    self.pose_adjustment[:, 1, :3] - self.pose_adjustment[:, 0, :3]).norm()
            metrics_dict["camera_opt_trajectory_rotation"] = (
                    self.pose_adjustment[:, 1, 3:] - self.pose_adjustment[:, 0, 3:]).norm()
            metrics_dict["camera_opt_translation"] = 0
            metrics_dict["camera_opt_rotation"] = 0
            for i in range(self.num_control_knots):
                metrics_dict["camera_opt_translation"] += self.pose_adjustment[:, i, :3].norm()
                metrics_dict["camera_opt_rotation"] += self.pose_adjustment[:, i, 3:].norm()
