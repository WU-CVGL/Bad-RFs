<h1 align=center font-weight:100> 😈<strong><i>BAD-RFs</i></strong>: <strong><i>B</i></strong>undle-<strong><i>ad</i></strong>justed <strong><i>R</i></strong>adience <strong><i>F</i></strong>ields from degraded images with continuous-time motion models</h1>

This repo contains:
- An implementation of our arXiv 2024 paper [**BAD-Gaussians**: Bundle Adjusted Deblur Gaussian Splatting](https://lingzhezhao.github.io/BAD-Gaussians/),
- An accelerated reimplementation of our CVPR 2023 paper [**BAD-NeRF**: Bundle Adjusted Deblur Neural Radiance Fields](https://wangpeng000.github.io/BAD-NeRF/),

based on the [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) framework.

In the future, we will continue to explore *bundle-adjusted radience fields*, add more accelerated implementations
to this repo, such as a reimplementation of our ICLR 2024 paper [**USB-NeRF**: Unrolling Shutter Bundle Adjusted Neural Radiance Fields](https://arxiv.org/abs/2310.02687).

## Demo

Deblurring & novel-view synthesis results on [Deblur-NeRF](https://github.com/limacv/Deblur-NeRF/)'s real-world motion-blurred data:

<video src="https://github.com/WU-CVGL/Bad-RFs/assets/43722188/d0ff1c69-1c7c-4ac2-bcf4-d625f95e06bd"></video>

> Left: BAD-Gaussians deblured novel-view renderings;
>
> Right: Input images.


## Quickstart

### 1. Installation

You may check out the original [`nerfstudio`](https://github.com/nerfstudio-project/nerfstudio) repo for prerequisites and dependencies. 
Currently, our codebase is build on top of the latest version of nerfstudio (v1.0.2),
so if you have an older version of nerfstudio installed,
please `git clone` the main branch and install the latest version.

TL;DR: You can install `nerfstudio` with:

```sh
# (Optional) create a fresh conda env
conda create --name nerfstudio -y python=3.10
conda activate nerfstudio

# install dependencies
pip install --upgrade pip setuptools
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# install nerfstudio!
git clone https://github.com/nerfstudio-project/nerfstudio
cd nerfstudio
pip install -e .
```

Besides, we use [pypose](https://github.com/pypose/pypose) to implement the pose interpolation. You can install it with:

```sh
pip install pypose
```

Then you can clone and install this repo as a python package with:

```sh
git clone https://github.com/WU-CVGL/Bad-RFs
cd Bad-RFs
pip install -e .
```

### 2. Prepare the dataset

#### Deblur-NeRF Synthetic Dataset (Re-rendered)

As described in the previous BAD-NeRF paper, we re-rendered Deblur-NeRF\'s synthetic dataset with 51 interpolations per blurry image.

Additionally, in the previous BAD-NeRF paper, we directly run COLMAP on blurry images only, with neither ground-truth 
camera intrinsics nor sharp novel-view images. We find this is quite challenging for COLMAP - it may fail to 
reconstruct the scene and we need to re-run COLMAP for serval times. To this end, we provided a new set of data, 
where we ran COLMAP with ground-truth camera intrinsics over both blurry and sharp novel-view images, 
named `bad-nerf-gtK-colmap-nvs`:

[Download link](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EoCe3vaC9V5Fl74DjbGriwcBKj1nbB0HQFSWnVTLX7qT9A)

#### Deblur-NeRF Real Dataset

You can directly download the `real_camera_motion_blur` folder from [Deblur-NeRF](https://limacv.github.io/deblurnerf/).

#### Your Custom Dataset

1. Use the [`ns-process-data` tool from Nerfstudio](https://docs.nerf.studio/reference/cli/ns_process_data.html)
    to process deblur-nerf training images. 

    For example, if the
    [dataset from BAD-NeRF](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EsgdW2cRic5JqerhNbTsxtkBqy9m6cbnb2ugYZtvaib3qA?e=bjK7op)
    is in `llff_data`, execute:

    ```
    ns-process-data images \
        --data llff_data/blurtanabata/images \
        --output-dir data/my_data/blurtanabata
    ```

2. The folder `data/my_data/blurtanabata` is ready.
> Note: Although nerfstudio does not model the NDC scene contraction for LLFF data, 
> we found that `scale_factor = 0.25` works well on LLFF datasets.
> If your data is captured in a [LLFF fashion](https://github.com/Fyusion/LLFF#using-your-own-input-images-for-view-synthesis) (i.e. forward-facing), 
> instead of object-centric like Mip-NeRF 360, 
> you can pass the `scale_factor = 0.25` parameter to the nerfstudio dataparser (which is already set to default in our `DeblurNerfDataParser`),
> e.g., `ns-train bad-nerfacto --data data/my_data/my_seq --vis viewer+tensorboard nerfstudio-data --scale_factor 0.25`

### 3. Training

#### BAD-Gaussians

For `Deblur-NeRF synthetic` dataset, train with:

```sh
ns-train bad-gaussians \
    --data data/bad-nerf-gtK-colmap-nvs/blurtanabata \
    --vis viewer+tensorboard \
    deblur-nerf-data
```

For `Deblur-NeRF real` dataset with `downscale_factor=4`, train with:
```sh
ns-train bad-gaussians \
    --data data/real_camera_motion_blur/blurdecoration \
    --pipeline.model.camera-optimizer.mode "cubic" \
    --vis viewer+tensorboard \
    deblur-nerf-data \
    --downscale_factor 4
```

For `Deblur-NeRF real` dataset with full resolution, train with:
```sh
ns-train bad-gaussians \
    --data data/real_camera_motion_blur/blurdecoration \
    --pipeline.model.camera-optimizer.mode "cubic" \
    --pipeline.model.camera-optimizer.num_virtual_views 15 \
    --pipeline.model.num_downscales 2 \
    --pipeline.model.resolution_schedule 3000 \
    --vis viewer+tensorboard \
    deblur-nerf-data
```

For custom data processed with `ns-process-data`, train with:

```bash
ns-train bad-gaussians \
    --data data/my_data/blurtanabata \
    --vis viewer+tensorboard \
    nerfstudio-data --eval_mode "all"
```

#### BAD-nerfacto

For `Deblur-NeRF synthetic` dataset and `Deblur-NeRF real` dataset, train with:

```sh
ns-train bad-nerfacto \
    --data data/bad-nerf-gtK-colmap-nvs/blurtanabata \
    --vis viewer+tensorboard \
    deblur-nerf-data
```

```bash
ns-train bad-nerfacto \
    --pipeline.model.camera-optimizer.mode "cubic" \
    --pipeline.model.camera-optimizer.num_virtual_views 15 \
    --data data/real_camera_motion_blur/blurdecoration \
    --vis viewer+tensorboard \
    deblur-nerf-data
```

For custom data processed with `ns-process-data`, train with:

```bash
ns-train bad-nerfacto \
    --data data/my_data/blurtanabata \
    --vis viewer+tensorboard \
    image-restore-data
```

### 4. Render videos

```bash
ns-render interpolate \
  --load-config outputs/tanabata/bad-gaussians/<your_experiment_date_time>/config.yml \
  --pose-source train \
  --frame-rate 30 \
  --interpolation-steps 10 \
  --output-path renders/<your_filename>.mp4
```

> Note1: You can add the `--render-nearest-camera True` option to compare with the blurry inputs, but it will slow down the rendering process significantly.
>
> Note2: The working directory when executing this command must be the parent of `outputs`, i.e. the same directory when training.
>
> Note3: You can find more information of this command in the [nerfstudio docs](https://docs.nerf.studio/reference/cli/ns_render.html#ns-render).

### 5. Debug with your IDE

Open this repo with your IDE, create a configuration, and set the executing python script path to
`<nerfstudio_path>/nerfstudio/scripts/train.py`, with the parameters above.


## Citation

If you find this useful, please consider citing:

```bibtex
@misc{zhao2024badgaussians,
    title={{BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting}},
    author={Zhao, Lingzhe and Wang, Peng and Liu, Peidong},
    year={2024},
    eprint={2403.11831},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}

@software{zhao2023badrfs,
    title     = {{Bad-RFs: Bundle-adjusted Radiance Fields from Degraded Images with Continuous-time Motion Models}},
    author    = {Zhao, Lingzhe and Wang, Peng and Liu, Peidong},
    year      = {2023},
    url       = {{https://github.com/WU-CVGL/Bad-RFs}}
}

@InProceedings{wang2023badnerf,
    title     = {{BAD-NeRF: Bundle Adjusted Deblur Neural Radiance Fields}},
    author    = {Wang, Peng and Zhao, Lingzhe and Ma, Ruijie and Liu, Peidong},
    month     = {June},
    year      = {2023},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    pages     = {4170-4179}
}
```

## Acknowledgment

- Kudos to the [Nerfstudio](https://github.com/nerfstudio-project/) contributors for their amazing work:

```bibtex
@inproceedings{nerfstudio,
	title        = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
	author       = {
		Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi, Brent
		and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
		Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa,
		Angjoo
	},
	year         = 2023,
	booktitle    = {ACM SIGGRAPH 2023 Conference Proceedings},
	series       = {SIGGRAPH '23}
}

@software{Ye_gsplat,
    author  = {Ye, Vickie and Turkulainen, Matias, and the Nerfstudio team},
    title   = {{gsplat}},
    url     = {https://github.com/nerfstudio-project/gsplat}
}

@misc{ye2023mathematical,
    title={Mathematical Supplement for the $\texttt{gsplat}$ Library}, 
    author={Vickie Ye and Angjoo Kanazawa},
    year={2023},
    eprint={2312.02121},
    archivePrefix={arXiv},
    primaryClass={cs.MS}
}
```
