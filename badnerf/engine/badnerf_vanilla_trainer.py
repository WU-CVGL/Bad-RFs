"""
BAD-NeRF vanilla trainer.
"""

import functools
import os
from dataclasses import dataclass, field
from typing import Type

import cv2
import torch

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.writer import EventName, TimeWriter, to8b

from badnerf.pipelines.badnerf_vanilla_pipeline import BadNerfVanillaPipeline, BadNerfVanillaPipelineConfig

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


@dataclass
class BadNerfVanillaTrainerConfig(TrainerConfig):
    """Configuration for BAD-NeRF training"""

    _target: Type = field(default_factory=lambda: BadNerfVanillaTrainer)
    pipeline: BadNerfVanillaPipelineConfig = BadNerfVanillaPipelineConfig()
    """BAD-NeRF pipeline configuration"""


class BadNerfVanillaTrainer(Trainer):
    """BAD-NeRF Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
        training_state: Current model training state.
    """

    config: BadNerfVanillaTrainerConfig
    pipeline: BadNerfVanillaPipeline

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            metrics_dict, images_dicts = self.pipeline.get_average_eval_metrics_and_images(step=step)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)

            writer_log_path = self.base_dir / self.config.logging.relative_log_dir
            image_dir = writer_log_path / f"{step:06}"
            if not image_dir.exists():
                image_dir.mkdir(parents=True)

            for image_idx, image_dict in images_dicts.items():
                for image_name, image_data in image_dict.items():
                    data = image_data.detach().cpu()
                    filename = f"{image_name}_{image_idx:04}"
                    if "rgb" in image_name or "blur" == image_name or "gt" == image_name:
                        path = str((image_dir / f"{filename}.png").resolve())
                        cv2.imwrite(path, cv2.cvtColor(to8b(data).numpy(), cv2.COLOR_RGB2BGR))
                    else:
                        path = str((image_dir / f"{filename}.exr").resolve())
                        cv2.imwrite(path, data.numpy())
