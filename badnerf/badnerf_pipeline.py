"""BAD-NeRF pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Optional, Type

import torch
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from nerfstudio.utils.writer import to8b

from badnerf.badnerfacto import BadNerfactoModel, BadNerfactoModelConfig
from badnerf.badnerf_datamanager import BadNerfDataManager, BadNerfDataManagerConfig

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2


@dataclass
class BadNerfPipelineConfig(VanillaPipelineConfig):
    """BAD-NeRF pipeline config"""

    _target: Type = field(default_factory=lambda: BadNerfPipeline)
    """The target class to be instantiated."""
    datamanager: BadNerfDataManagerConfig = field(default_factory=BadNerfDataManagerConfig)
    """specifies the datamanager config"""
    model: BadNerfactoModelConfig = field(default_factory=BadNerfactoModelConfig)
    """specifies the model config"""
    eval_render_start_end: bool = False
    """whether to render and save the starting and ending virtual sharp images in eval"""
    eval_render_reblur: bool = False
    """whether to render and save the re-blurred images with learned trajectory in eval. Note: Slow & VRAM hungry!
    Reduce VRAM consumption by passing argument `--pipeline.model.eval_num_rays_per_chunk=16384` or less.
    """


class BadNerfPipeline(VanillaPipeline):
    """BAD-NeRF pipeline"""

    config: BadNerfPipelineConfig
    model: BadNerfactoModel
    datamanager: BadNerfDataManager

    @profiler.time_function
    def get_average_eval_image_metrics(
            self, step: Optional[int] = None, output_path: Optional[Path] = None, get_std: bool = False
    ):
        """Iterate over all the images in the eval dataset and get the average.

        Also saves the rendered images to disk if output_path is provided.

        Args:
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        render_list = ["mid"]
        if self.config.eval_render_start_end:
            render_list += ["start", "end"]
        if self.config.eval_render_reblur:
            render_list += ["uniform"]
        assert isinstance(self.datamanager, (VanillaDataManager, ParallelDataManager, FullImageDatamanager))
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                image_idx = batch['image_idx']
                images_dict = {
                    f"{image_idx:04}_blur": batch["blur"][:, :, :3],
                    f"{image_idx:04}_gt": batch["image"][:, :, :3],
                }
                for mode in render_list:
                    outputs = self.model.get_outputs_for_camera(camera, mode=mode)
                    for key, value in outputs.items():
                        if "uniform" == mode:
                            filename = f"{image_idx:04}_reblur"
                        else:
                            filename = f"{image_idx:04}_{key}_{mode}"
                        if "rgb" in key:
                            images_dict[filename] = value
                        if "depth" in key and "uniform" != mode:
                            images_dict[filename] = value
                    if "mid" == mode:
                        metrics_dict, _ = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    image_dir = output_path / f"{step:06}"
                    if not image_dir.exists():
                        image_dir.mkdir(parents=True)
                    for filename, data in images_dict.items():
                        data = data.detach().cpu()
                        if "rgb" in filename or "blur" in filename or "gt" in filename:
                            path = str((image_dir / f"{filename}.png").resolve())
                            cv2.imwrite(path, cv2.cvtColor(to8b(data).numpy(), cv2.COLOR_RGB2BGR))
                        else:
                            path = str((image_dir / f"{filename}.exr").resolve())
                            cv2.imwrite(path, data.numpy())

                assert "num_rays_per_sec" not in metrics_dict
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
        self.train()
        return metrics_dict
