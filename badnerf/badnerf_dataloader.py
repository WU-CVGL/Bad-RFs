"""
BAD-NeRF dataloaders.
"""

from typing import Dict, Optional, Tuple, Union

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader, FixedIndicesEvalDataloader
from nerfstudio.utils.misc import get_dict_to_torch


class BADNeRFRandIndicesEvalDataloader(RandIndicesEvalDataloader):
    """eval_dataloader that returns random images.

    Args:
        input_dataset: sharp GT image
        blurry_dataset: corresponding blurry input image
        eval_image_ray_generator: Ray generator of BAD-NeRF
        device: Device to load data to
    """

    def __init__(
            self,
            input_dataset: InputDataset,
            blurry_dataset: InputDataset,
            device: Union[torch.device, str] = "cpu",
            **kwargs,
    ):
        super().__init__(input_dataset, device, **kwargs)
        self.blurry_dataset = blurry_dataset

    def get_camera(self, image_idx: int = 0) -> Tuple[Cameras, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        camera = self.cameras[image_idx : image_idx + 1]
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        batch["blur"] = self.blurry_dataset[image_idx]["image"]
        assert isinstance(batch, dict)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, batch


class BADNeRFFixedIndicesEvalDataloader(FixedIndicesEvalDataloader):
    """fixed_indices_eval_dataloader that returns a fixed set of indices.

    Args:
        input_dataset: sharp GT image
        blurry_dataset: corresponding blurry input image
        eval_image_ray_generator: Ray generator of BAD-NeRF
        image_indices: List of image indices to load data from. If None, then use all images.
        device: Device to load data to
    """

    def __init__(
            self,
            input_dataset: InputDataset,
            blurry_dataset: InputDataset,
            image_indices: Optional[Tuple[int]] = None,
            device: Union[torch.device, str] = "cpu",
            **kwargs,
    ):
        super().__init__(input_dataset, image_indices, device, **kwargs)
        self.blurry_dataset = blurry_dataset

    def get_camera(self, image_idx: int = 0) -> Tuple[Cameras, Dict]:
        """Returns the data for a specific image index.

        Args:
            image_idx: Camera image index
        """
        camera = self.cameras[image_idx : image_idx + 1]
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        batch["blur"] = self.blurry_dataset[image_idx]["image"]
        assert isinstance(batch, dict)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, batch
