"""
BAD-NeRF vanilla pipeline.
"""

from dataclasses import dataclass, field
from typing import Type, Optional

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler

import badnerf.pipelines.badnerf_pipeline_common as _common


@dataclass
class BadNerfPipelineConfig(VanillaPipelineConfig):
    """BAD-NeRF pipeline config"""

    _target: Type = field(default_factory=lambda: BadNerfPipeline)
    num_virtual_views: int = 10
    """Number of virtual sharp images to re-blur"""


class BadNerfPipeline(VanillaPipeline):
    """BAD-NeRF pipeline"""

    config: BadNerfPipelineConfig

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        metrics_dict, _ = _common.get_average_eval_metrics_and_images(self, step)
        return metrics_dict

    @profiler.time_function
    def get_average_eval_metrics_and_images(self, step: Optional[int] = None):
        """Iterate over all the images in the eval dataset and get the average metrics & images.

        Returns:
            metrics_dict: dictionary of metrics
            images_dict: dictionary of images
        """
        metrics_dict, images_dicts = _common.get_average_eval_metrics_and_images(self, step)
        return metrics_dict, images_dicts
