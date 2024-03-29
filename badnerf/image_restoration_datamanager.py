"""
Image restoration datamanager.
"""
from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE

from badnerf.image_restoration_dataloader import ImageRestorationFixedIndicesEvalDataloader, ImageRestorationRandIndicesEvalDataloader


@dataclass
class ImageRestorationDataManagerConfig(VanillaDataManagerConfig):
    """A depth datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: ImageRestorationDataManager)


class ImageRestorationDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for BAD-NeRF

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ImageRestorationDataManagerConfig

    def setup_eval(self):
        """Sets up the data loader for evaluation"""
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
        # for loading full images
        self.fixed_indices_eval_dataloader = ImageRestorationFixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            degraded_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = ImageRestorationRandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            degraded_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
