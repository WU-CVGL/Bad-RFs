'''Viewer of BAD-RFs'''
import numpy as np
import torch
import viser.transforms as vtf

from nerfstudio.viewer.viewer import Viewer, VISER_NERFSTUDIO_SCALE_RATIO

from badrfs.bad_camera_optimizer import BadCameraOptimizer


class BadRfViewer(Viewer):
    # BAD-RFs: Overriding original update_camera_poses because BadNerfCameraOptimizer returns LieTensor

    def update_camera_poses(self):
        # TODO this fn accounts for like ~5% of total train time
        # Update the train camera locations based on optimization
        assert self.camera_handles is not None
        if hasattr(self.pipeline.datamanager, "train_camera_optimizer"):
            camera_optimizer = self.pipeline.datamanager.train_camera_optimizer
        elif hasattr(self.pipeline.model, "camera_optimizer"):
            camera_optimizer = self.pipeline.model.camera_optimizer
        else:
            return
        idxs = list(self.camera_handles.keys())
        with torch.no_grad():
            assert isinstance(camera_optimizer, BadCameraOptimizer)
            for i, key in enumerate(idxs):
                # both are numpy arrays
                c2w_orig = self.original_c2w[key]
                c2w = camera_optimizer.apply_to_c2w(torch.tensor(c2w_orig), key).cpu().numpy()
                R = vtf.SO3.from_matrix(c2w[:3, :3])  # type: ignore
                R = R @ vtf.SO3.from_x_radians(np.pi)
                self.camera_handles[key].position = c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
                self.camera_handles[key].wxyz = R.wxyz
