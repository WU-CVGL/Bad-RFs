"""
BAD-NeRF config.
"""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from badnerf.cameras.badnerf_camera_optimizer import BadNerfCameraOptimizerConfig
from badnerf.data.badnerf_datamanager import BadNerfDataManagerConfig
from badnerf.data.badnerf_dataparser import BadNerfDataParserConfig
from badnerf.engine.badnerf_trainer import BadNerfTrainerConfig
from badnerf.models.badnerfacto import BadNerfactoModelConfig
from badnerf.pipelines.badnerf_pipeline import BadNerfPipelineConfig

badnerf_nerfacto = MethodSpecification(
    config=BadNerfTrainerConfig(
        method_name="bad-nerfacto",
        steps_per_eval_all_images=500,
        steps_per_save=2000,
        max_num_iterations=30001,
        mixed_precision=False,
        use_grad_scaler=True,
        pipeline=BadNerfPipelineConfig(
            datamanager=BadNerfDataManagerConfig(
                dataparser=BadNerfDataParserConfig(),
                train_num_rays_per_batch=1024,
                eval_num_rays_per_batch=1024,
            ),
            model=BadNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                camera_optimizer=BadNerfCameraOptimizerConfig(mode="linear", num_virtual_views=10),
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=5e-4, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-6, max_steps=10000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Implementation of BAD-NeRF based on nerfacto",
)
