"""
BAD-Gaussians model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Type, Union

import torch

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat
from nerfstudio.model_components import renderers

from badrfs.bad_camera_optimizer import (
    BadCameraOptimizer,
    BadCameraOptimizerConfig,
    TrajSamplingMode,
)
from badrfs.badrf_losses import EdgeAwareVariationLoss
from badrfs.badrf_strategy import BadDefaultStrategy


@dataclass
class BadGaussiansModelConfig(SplatfactoModelConfig):
    """BAD-Gaussians Model config"""

    _target: Type = field(default_factory=lambda: BadGaussiansModel)
    """The target class to be instantiated."""

    rasterize_mode: Literal["classic", "antialiased"] = "antialiased"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    Refs:
    1. https://github.com/nerfstudio-project/gsplat/pull/117
    2. https://github.com/nerfstudio-project/nerfstudio/pull/2888
    3. Yu, Zehao, et al. "Mip-Splatting: Alias-free 3D Gaussian Splatting." arXiv preprint arXiv:2311.16493 (2023).
    """

    camera_optimizer: BadCameraOptimizerConfig = field(default_factory=BadCameraOptimizerConfig)
    """Config of the camera optimizer to use"""

    cull_alpha_thresh: float = 0.005
    """Threshold for alpha to cull gaussians. Default: 0.1 in splatfacto, 0.005 in splatfacto-big."""

    densify_grad_thresh: float = 4e-4
    """[IMPORTANT] Threshold for gradient to densify gaussians. Default: 4e-4. Tune it smaller with complex scenes."""

    continue_cull_post_densification: bool = False
    """Whether to continue culling after densification. Default: True in splatfacto, False in splatfacto-big."""

    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled.
    Default: 250. Use 3000 with high resolution images (e.g. higher than 1920x1080).
    """

    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number. Default: 0. Use 2 with high resolution images."""

    tv_loss_lambda: Optional[float] = None
    """weight of total variation loss"""


class BadGaussiansModel(SplatfactoModel):
    """BAD-Gaussians Model

    Args:
        config: configuration to instantiate model
    """

    config: BadGaussiansModelConfig
    camera_optimizer: BadCameraOptimizer

    def __init__(self, config: BadGaussiansModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        # Scale densify_grad_thresh by the number of virtual views
        self.config.densify_grad_thresh /= self.config.camera_optimizer.num_virtual_views
        # (Experimental) Total variation loss
        self.tv_loss = EdgeAwareVariationLoss(in1_nc=3)

    def populate_modules(self) -> None:
        super().populate_modules()
        self.camera_optimizer: BadCameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        if hasattr(self.config, "strategy") and self.config.strategy == "default":
            self.strategy = BadDefaultStrategy(
                prune_opa=self.config.cull_alpha_thresh,
                grow_grad2d=self.config.densify_grad_thresh,
                grow_scale3d=self.config.densify_size_thresh,
                grow_scale2d=self.config.split_screen_size,
                prune_scale3d=self.config.cull_scale_thresh,
                prune_scale2d=self.config.cull_screen_size,
                refine_scale2d_stop_iter=self.config.stop_screen_size_at,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                reset_every=self.config.reset_alpha_every * self.config.refine_every,
                refine_every=self.config.refine_every,
                pause_refine_after_reset=self.num_train_data + self.config.refine_every,
                absgrad=self.config.use_absgrad,
                revised_opacity=False,
                verbose=True,
                num_virtual_views=self.config.camera_optimizer.num_virtual_views,
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

    def forward(
            self,
            camera: Cameras,
            mode: TrajSamplingMode = "uniform",
    ) -> Dict[str, Union[torch.Tensor, List]]:
        return self.get_outputs(camera, mode)

    def get_outputs(
            self, camera: Cameras,
            mode: TrajSamplingMode = "uniform",
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Camera and returns a dictionary of outputs.

        Args:
            camera: Input camera. This camera should have all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        is_training = self.training and torch.is_grad_enabled()
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"

        # BAD-RFs: get virtual cameras
        virtual_viewmats = self.camera_optimizer.apply_to_camera(camera, mode).squeeze()
        if virtual_viewmats.dim() == 2:
            virtual_viewmats = virtual_viewmats.unsqueeze(0)
        assert virtual_viewmats.dim() == 3, "viewmats should be 3D tensor"
        # viewmats = pp.mat2SE3(virtual_viewmats).Inv().matrix()
        viewmats = get_viewmat(virtual_viewmats)
        # (..., 3, 4) -> (..., 4, 4)
        if viewmats.shape[-2] == 3:
            viewmats = torch.cat([viewmats, torch.tensor([[[0, 0, 0, 1]]], device=self.device).expand(viewmats.shape[0], -1, -1)], dim=1)

        if is_training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            # logic for setting the background of the scene
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        K = camera.get_intrinsics_matrices().cuda()
        Ks = K.tile(viewmats.shape[0], 1, 1)
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop)
            sh_degree_to_use = None

        # BAD-RFs: render virtual views
        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmats,  # [num_virtual_views, 4, 4]
            Ks=Ks,  # [num_virtual_views, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.config.use_absgrad,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        alpha = alpha[:, ...]
        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        # BAD-RFs: average the virtual views
        rgb = rgb.mean(0)[None]

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

    @torch.no_grad()
    def get_outputs_for_camera(
            self,
            camera: Cameras,
            obb_box: Optional[OrientedBox] = None,
            mode: TrajSamplingMode = "mid",
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        # BAD-RFs: camera.to(device) will drop metadata
        metadata = camera.metadata
        camera = camera.to(self.device)
        camera.metadata = metadata
        outs = self.get_outputs(camera, mode=mode)
        return outs  # type: ignore

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        # Add total variation loss
        rgb = outputs["rgb"].permute(2, 0, 1).unsqueeze(0)  # H, W, 3 to 1, 3, H, W
        if self.config.tv_loss_lambda is not None:
            loss_dict["tv_loss"] = self.tv_loss(rgb) * self.config.tv_loss_lambda
        # Add loss from camera optimizer
        self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        # Add metrics from camera optimizer
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        param_groups = super().get_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups
