"""
Implementation of BAD-NeRF based on nerfacto.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils import colormaps

from badnerf.badnerf_camera_optimizer import (
    BadNerfCameraOptimizer,
    BadNerfCameraOptimizerConfig,
    TrajSamplingMode,
)


@dataclass
class BadNerfactoModelConfig(NerfactoModelConfig):
    """BAD-NeRF-nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: BadNerfactoModel)
    """The target class to be instantiated."""

    camera_optimizer: BadNerfCameraOptimizerConfig = field(default_factory=BadNerfCameraOptimizerConfig)
    """Config of the camera optimizer to use"""


class BadNerfactoModel(NerfactoModel):
    """BAD-NeRF-nerfacto Model

    Args:
        config: configuration to instantiate model
    """

    config: BadNerfactoModelConfig
    camera_optimizer: BadNerfCameraOptimizer

    def __init__(self, config: BadNerfactoModelConfig, **kwargs) -> None:
        super().__init__(config=config, **kwargs)

    def forward(
            self,
            ray_bundle: RayBundle,
            mode: TrajSamplingMode = "uniform",
    ) -> Dict[str, Union[torch.Tensor, List]]:
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
        return self.get_outputs(ray_bundle, mode)

    def get_outputs(
            self,
            ray_bundle: RayBundle,
            mode: TrajSamplingMode = "uniform",
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.
        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.
            mode: Trajectory sampling mode for BadNerfCameraOptimizer.
        Returns:
            Outputs of model. (ie. rendered colors)
        """
        # apply the camera optimizer pose tweaks
        ray_bundle = self.camera_optimizer.apply_to_raybundle(ray_bundle, mode)
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        if mode == "uniform":
            # BAD-NeRF: synthesize blurry rgb
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
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
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
        blur = batch["degraded"].to(self.device)
        rgb = outputs["rgb"]
        if "accumulation" in outputs:
            accumulation = outputs["accumulation"]
            acc = colormaps.apply_colormap(outputs["accumulation"])
            combined_acc = torch.cat([acc], dim=1)
        else:
            accumulation = None

        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=accumulation,
        )
        combined_rgb = torch.cat([blur, rgb, gt], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt = torch.moveaxis(gt, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt, rgb)
        ssim = self.ssim(gt, rgb)
        lpips = self.lpips(gt, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}
        images_dict = {"img": combined_rgb, "depth": combined_depth}
        if "accumulation" in outputs:
            images_dict["accumulation"] = combined_acc

        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera(
            self,
            camera: Cameras,
            obb_box: Optional[OrientedBox] = None,
            mode: TrajSamplingMode = "mid",
    ) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model."""
        raybundle = camera.generate_rays(camera_indices=0, keep_shape=True, obb_box=obb_box)

        # Set camera indices for the camera optimizer to get the correct camera pose adjustments
        if camera.metadata is not None:
            raybundle.set_camera_indices(camera.metadata["cam_idx"])

        image_height, image_width = camera[0].height.item(), camera[0].width.item()
        return self.get_outputs_for_camera_ray_bundle(raybundle, mode, image_height, image_width)

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(
            self,
            camera_ray_bundle: RayBundle,
            mode: TrajSamplingMode = "mid",
            image_height: int = None,
            image_width: int = None,
    ) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model."""
        input_device = camera_ray_bundle.directions.device
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        if image_height is None or image_width is None:
            image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            # move the chunk inputs to the model device
            ray_bundle = ray_bundle.to(self.device)
            outputs = self.forward(ray_bundle, mode)
            for output_name, output in outputs.items():  # type: ignore
                if not isinstance(output, torch.Tensor):
                    # TODO: handle lists of tensors as well
                    continue
                # move the chunk outputs from the model device back to the device of the inputs.
                outputs_lists[output_name].append(output.to(input_device))
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs
