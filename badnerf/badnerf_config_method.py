"""
BAD-NeRF config.
"""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from badnerf.badnerf_camera_optimizer import BadNerfCameraOptimizerConfig
from badnerf.image_restoration_datamanager import ImageRestorationDataManagerConfig
from badnerf.image_restoration_dataparser import ImageRestorationDataParserConfig
from badnerf.image_restoration_full_image_datamanager import ImageRestorationFullImageDataManagerConfig
from badnerf.image_restoration_trainer import ImageRestorationTrainerConfig
from badnerf.bad_gaussians import BadGaussiansModelConfig
from badnerf.badnerfacto import BadNerfactoModelConfig
from badnerf.image_restoration_pipeline import ImageRestorationPipelineConfig


badnerf_nerfacto = MethodSpecification(
    config=ImageRestorationTrainerConfig(
        method_name="bad-nerfacto",
        steps_per_eval_batch=500,
        steps_per_eval_image=500,
        steps_per_eval_all_images=2000,
        steps_per_save=2000,
        max_num_iterations=30001,
        mixed_precision=False,
        use_grad_scaler=True,
        pipeline=ImageRestorationPipelineConfig(
            datamanager=ImageRestorationDataManagerConfig(
                dataparser=ImageRestorationDataParserConfig(),
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

bad_gaussians = MethodSpecification(
    config=ImageRestorationTrainerConfig(
        method_name="bad-gaussian",
        steps_per_eval_image=100,
        steps_per_eval_batch=100,
        steps_per_save=2000,
        steps_per_eval_all_images=500,
        max_num_iterations=30001,
        mixed_precision=False,
        use_grad_scaler=False,
        gradient_accumulation_steps={"camera_opt": 25},
        pipeline=ImageRestorationPipelineConfig(
            eval_render_start_end=True,
            eval_render_estimated=True,
            datamanager=ImageRestorationFullImageDataManagerConfig(
                dataparser=ImageRestorationDataParserConfig(
                    load_3D_points=True,
                    eval_mode="interval",
                    eval_interval=8,
                ),
            ),
            model=BadGaussiansModelConfig(
                camera_optimizer=BadNerfCameraOptimizerConfig(mode="linear", num_virtual_views=10),
                use_scale_regularization=True,
            ),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5,
                    max_steps=30000,
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Implementation of BAD-Gaussians",
)
