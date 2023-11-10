"""
Implementation of BAD-NeRF based on nerfacto.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps

from badnerf.cameras.badnerf_camera_optimizer import BadNerfCameraOptimizer, BadNerfCameraOptimizerConfig


@dataclass
class BadNerfactoModelConfig(NerfactoModelConfig):
    """BAD-NeRF-nerfacto Model Config"""

    _target: Type = field(
        default_factory=lambda: BadNerfactoModel
    )
    """The target class to be instantiated."""

    camera_optimizer: BadNerfCameraOptimizerConfig
    """Config of the camera optimizer to use"""


class BadNerfactoModel(NerfactoModel):
    """BAD-NeRF-nerfacto Model model

    Args:
        config: configuration to instantiate model
    """

    config: BadNerfactoModelConfig
    camera_optimizer: BadNerfCameraOptimizer

    def __init__(self, config: BadNerfactoModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def get_outputs(self, ray_bundle: RayBundle):
        is_training = self.training and torch.is_grad_enabled()
        # apply the camera optimizer pose tweaks
        ray_bundle = self.camera_optimizer.apply_to_raybundle(ray_bundle, is_training)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        # synthesize blurry rgb
        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        if is_training:
            n = self.camera_optimizer.config.num_virtual_views
            s = ray_bundle.origins.shape[0] // n
            rgb = rgb.view(s, n, 3).mean(dim=1)

        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            # "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if is_training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if is_training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_image_metrics_and_images(
            self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        metrics_dict, images_dict = self.get_badnerf_eval_image_metrics_and_images(outputs, batch)

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_badnerf_eval_image_metrics_and_images(
            self,
            outputs: Dict[str, Tensor],
            batch: Dict[str, Tensor],
    ) -> Tuple[Dict[str, float], Dict[str, Tensor]]:
        """Parse the evaluation outputs.
        Args:
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt = batch["image"][:, :, :3].to(self.device)
        blur = batch["blur"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([blur, rgb, gt], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt = torch.moveaxis(gt, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt, rgb)
        ssim = self.ssim(gt, rgb)
        lpips = self.lpips(gt, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        return metrics_dict, images_dict
