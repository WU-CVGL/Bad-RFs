"""
Pose and Intrinsics Optimizers
"""

from __future__ import annotations

import functools
from typing import Literal, Optional, Tuple, Type, Union

import pypose as pp
import torch
from dataclasses import dataclass, field
from jaxtyping import Float, Int
from pypose import LieTensor
from torch import Tensor, nn
from typing_extensions import assert_never

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig

from badnerf.cameras.spline import linear_interpolation, linear_interpolation_mid


@dataclass
class BadNerfCameraOptimizerConfig(InstantiateConfig):
    """Configuration of BAD-NeRF camera optimizer."""

    _target: Type = field(default_factory=lambda: BadNerfCameraOptimizer)
    """The target class to be instantiated."""

    mode: Literal["off", "linear", "bspline"] = "off"
    """Pose optimization strategy to use.
    linear: linear interpolation on SE(3);
    bspline: cubic b-spline interpolation on SE(3)."""

    num_virtual_views: int = 10
    """The number of samples used to model the motion-blurring."""

    initial_noise_se3_std: float = 1e-5
    """Initial perturbation to pose delta on se(3). Must be non-zero to prevent NaNs."""


class BadNerfCameraOptimizer(nn.Module):
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
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices
        self.dof = 6
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
    ) -> Float[Tensor, "camera_indices self.num_control_knots self.dof"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        else:
            outputs.append(self.pose_adjustment.Exp()[indices.int(), :])

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
            if self.config.mode == "linear":
                return pp.identity_SE3(
                    *(indices.shape[0], self.num_control_knots),
                    device=self.pose_adjustment.device
                )
            else:
                assert_never(self.config.mode)
        return functools.reduce(pp.mul, outputs)

    def spline_interpolation_bundle(
            self, camera_indices: Int[Tensor, "num_rays"]
    ) -> Tuple[
        Float[LieTensor, "num_rays num_virtual_views 4"],
        Float[Tensor, "num_rays num_virtual_views 3"]
    ]:
        """
        Interpolate camera poses for each ray in the bundle.
        """
        camera_opt = self(camera_indices)
        camera_opt_to_camera_start = camera_opt[:, 0, :]
        camera_opt_to_camera_end = camera_opt[:, 1, :]
        q, t = linear_interpolation(
            camera_opt_to_camera_start,
            camera_opt_to_camera_end,
            torch.linspace(0, 1, self.config.num_virtual_views, device=camera_opt_to_camera_start.device)
        )
        return q, t

    def spline_interpolation_mid(
            self, camera_indices: Int[Tensor, "num_rays"]
    ) -> Tuple[
        Float[LieTensor, "num_rays 4"],
        Float[Tensor, "num_rays 3"]
    ]:
        """
        Get median camera poses for each ray in the bundle.
        """
        camera_opt = self(camera_indices)
        camera_opt_to_camera_start = camera_opt[:, 0, :]
        camera_opt_to_camera_end = camera_opt[:, 1, :]
        q, t = linear_interpolation_mid(
            camera_opt_to_camera_start, camera_opt_to_camera_end
        )
        return q, t

    def apply_to_raybundle(self, ray_bundle: RayBundle, training: bool) -> RayBundle:
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
        if self.config.mode == "linear":
            if training:
                q, t = self.spline_interpolation_bundle(camera_ids)
                q = torch.flatten(q, start_dim=0, end_dim=1)
                t = torch.flatten(t, start_dim=0, end_dim=1)
                ray_bundle = ray_bundle._apply_fn_to_fields(repeat_fn)
            else:
                q, t = self.spline_interpolation_mid(camera_ids)

            ray_bundle.origins = ray_bundle.origins + t
            ray_bundle.directions = pp.SO3(q) @ ray_bundle.directions
            return ray_bundle
        else:
            assert_never(self.config.mode)

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        pass

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

    def get_correction_matrices(self):
        """Get optimized pose correction matrices"""
        return self(torch.arange(0, self.num_cameras).long())

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        camera_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups["camera_opt"] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0
