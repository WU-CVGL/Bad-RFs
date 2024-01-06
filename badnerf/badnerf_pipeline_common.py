"""
Common components in BAD-NeRF pipelines.
"""

from time import time
from typing import Optional

import torch
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from nerfstudio.pipelines.base_pipeline import VanillaPipeline


def get_average_eval_metrics_and_images(pipeline: VanillaPipeline, step: Optional[int] = None):
    """Iterate over all the images in the eval dataset and get the average.

    Returns:
        metrics_dict: dictionary of metrics
        image_dicts: dictionary of images dictionaries
    """
    pipeline.eval()
    metrics_dict_list = []
    num_images = len(pipeline.datamanager.fixed_indices_eval_dataloader)
    images_dicts = {}
    with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
    ) as progress:
        task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
        for camera, batch in pipeline.datamanager.fixed_indices_eval_dataloader:
            # time this the following line
            inner_start = time()
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            height, width = camera.height, camera.width
            num_rays = height * width
            metrics_dict, _ = pipeline.model.get_image_metrics_and_images(outputs, batch)
            assert "num_rays_per_sec" not in metrics_dict
            metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
            fps_str = "fps"
            assert fps_str not in metrics_dict
            metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
            metrics_dict_list.append(metrics_dict)

            image_idx = batch['image_idx']
            images_dict = {
                'blur': batch['blur'],
                'gt': batch['image'][:, :, :3],
            }
            output_image_keys = [key for key in outputs.keys() if 'rgb' in key]
            # output_image_keys += [key for key in outputs.keys() if 'depth' in key]
            # output_image_keys += [key for key in outputs.keys() if 'accumulation' in key]

            for key in output_image_keys:
                images_dict[key] = outputs[key]

            images_dicts[image_idx] = images_dict
            progress.advance(task)
    # average the metrics list
    metrics_dict = {}
    for key in metrics_dict_list[0].keys():
        metrics_dict[key] = float(
            torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
        )
    pipeline.train()
    return metrics_dict, images_dicts
