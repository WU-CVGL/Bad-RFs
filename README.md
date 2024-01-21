<h1 align=center> 😈𝘽𝙖𝙙-𝙉𝙚𝙍𝙁𝙨: 𝘽undle-𝙖𝙙justed 𝙉𝙚𝙍𝙁𝙨 from degraded images with continuous-time motion models</h1>

This repo contains an accelerated reimplementation of our CVPR paper [**BAD-NeRF**: Bundle Adjusted Deblur Neural Radiance Fields](https://wangpeng000.github.io/BAD-NeRF/),
based on the [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) framework.

In the future, we will continue to explore *bundle-adjusted neural radience fields*, add more accelerated implementations
to this repo, such as a reimplementation of our ICLR paper [**USB-NeRF**: Unrolling Shutter Bundle Adjusted Neural Radiance Fields](https://arxiv.org/abs/2310.02687).

## Demo

Deblurring & novel-view synthesis results on [Deblur-NeRF](https://github.com/limacv/Deblur-NeRF/)'s real-world motion-blurred data:

https://github.com/WU-CVGL/BAD-NeRFstudio/assets/43722188/944a6016-6d6a-4609-b8e3-1e04f768d3dd

https://github.com/WU-CVGL/BAD-NeRFstudio/assets/43722188/dd87c08e-9428-45a4-a609-e26277be1b2e

https://github.com/WU-CVGL/BAD-NeRFstudio/assets/43722188/13949669-971c-4d2c-a1b9-bd7ea8d82147

https://github.com/WU-CVGL/BAD-NeRFstudio/assets/43722188/f45b7c47-148c-4a63-a992-66855245c5c0

> Left: BAD-NeRFacto deblured novel-view renderings;
>
> Right: Input images.


## Quickstart

### 1. Installation

You may check out the original [`nerfstudio`](https://github.com/nerfstudio-project/nerfstudio) repo for prerequisites and dependencies. 
Currently, our codebase is build on top of the latest version of nerfstudio (v1.0.0),
so if you have an older version of nerfstudio installed,
please `git clone` the main branch and install the latest version.

Besides, we use [pypose](https://github.com/pypose/pypose) to implement the pose interpolation. You can install it with:

```bash
pip install pypose
```

Then you can clone and install this repo as a python package with:

```bash
git clone https://github.com/WU-CVGL/Bad-NeRFs
cd Bad-NeRFs
pip install -e .
```

### 2. Prepare the dataset

How to convert a deblur-nerf dataset for this dataparser:

1. Use the [`ns-process-data` tool from Nerfstudio](https://docs.nerf.studio/reference/cli/ns_process_data.html)
    to process deblur-nerf training images. 
    
    For example, if the
    [dataset from BAD-NeRF](https://westlakeu-my.sharepoint.com/:f:/g/personal/cvgl_westlake_edu_cn/EsgdW2cRic5JqerhNbTsxtkBqy9m6cbnb2ugYZtvaib3qA?e=bjK7op)
    is in `llff_data`, execute:

    ```
    ns-process-data images --data llff_data/blurtanabata/images --output-dir data/bad-nerf/blurtanabata
    ```

2. Copy the testing images (ground truth sharp images) to the new folder

    ```
    cp llff_data/blurtanabata/images_test data/bad-nerf/blurtanabata/
    ```

3. The folder `data/bad-nerf/blurtanabata` is ready.

> Note1: If you do not have the testing images, e.g. when training with real-world data
> (like those in [Deblur-NeRF](https://limacv.github.io/deblurnerf/)), you can skip the step 2.
>
> Note2: In the BadNerfDataparser, since nerfstudio does not model the NDC scene contraction for LLFF data,
> we set `scale_factor = 0.25`, which makes nerfacto works well on LLFF datasets.
> If your data is not captured in a LLFF fashion (i.e. forward-facing), such as object-centric like Mip-NeRF 360,
> you can set the `scale_factor = 1.`

### 3. Training

```bash
ns-train bad-nerfacto --data data/bad-nerf/blurtanabata --vis viewer+tensorboard
```

### 4. Render videos

```bash
ns-render interpolate \
  --load-config outputs/tanabata/bad-nerfacto/<your_experiment_date_time>/config.yml \
  --render-nearest-camera True \
  --order-poses True \
  --output-path renders/<your_filename>.mp4
```

### 5. Debug with your IDE

Open this repo with your IDE, create a configuration, and set the executing python script path to
`<nerfstudio_path>/nerfstudio/scripts/train.py`, with the parameters above.

## Evaluation

### Image deblurring

| Model                          | Dataset      | PSNR↑           | SSIM↑             | LPIPS↓            |Train Time (steps@time)|
|--------------------------------|--------------|-----------------|-------------------|-------------------|-----------------------|
| BAD-NeRF (paper)               | Cozy2room    | `32.15`         | 0.9170            | 0.0547            | 200k@11h              |
| `bad-nerfacto`                 | Cozy2room    | 29.74 / 31.59   | 0.8983 / `0.9403` | 0.0910 / `0.0406` | 5k@200s / 30k@18min   |
| BAD-NeRF (paper)               | Factory      | 32.08           | 0.9105            | 0.1218            | 200k@11h              |
| `bad-nerfacto`                 | Factory      | 31.00 / `32.97` | 0.9008 / `0.9381` | 0.1358 / `0.0929` | 5k@200s / 30k@18min   |
| BAD-NeRF (paper)               | Pool         | 33.36           | 0.8912            | 0.0802            | 200k@11h              |
| `bad-nerfacto`                 | Pool         | 31.64 / `33.62` | 0.8554 / `0.9079` | 0.1250 / `0.0584` | 5k@200s / 30k@18min   |
| BAD-NeRF (paper)               | Tanabata     | 27.88           | 0.8642            | 0.1179            | 200k@11h              |
| `bad-nerfacto`                 | Tanabata     | 26.88 / `29.32` | 0.8524 / `0.9133` | 0.1450 / `0.0895` | 5k@200s / 30k@18min   |
| BAD-NeRF (paper)               | Trolley      | 29.25           | 0.8892            | 0.0833            | 200k@11h              |
| `bad-nerfacto`                 | Trolley      | 27.45 / `31.00` | 0.8675 / `0.9371` | 0.1222 / `0.0445` | 5k@200s / 30k@18min   |
| BAD-NeRF (paper)               | ArchViz-low  | `31.27`         | 0.9005            | 0.1503            | 200k@11h              |
| `bad-nerfacto`                 | ArchViz-low  | 26.70 / 27.03   | 0.8893 / `0.9046` | 0.1672 / `0.1267` | 5k@200s / 30k@18min   |
| BAD-NeRF (paper)               | ArchViz-high | `28.07`         | 0.8234            | 0.2460            | 200k@11h              |
| `bad-nerfacto`                 | ArchViz-high | 26.22 / 27.32   | 0.8649 / `0.8894` | 0.2504 / `0.2061` | 5k@200s / 30k@18min   |

> Tested with AMD Ryzen 7950X CPU + NVIDIA RTX 4090 GPU, on Manjaro Linux, with CUDA 12.1 and PyTorch 2.0.1.
> Train speed may vary with different configurations.

## Citation

If you find this useful, please consider citing:

```bibtex
@misc{zhao2023badnerfs,
    title     = {{Bad-NeRFs: Bundle-adjusted Neural Radiance Fields from Degraded Images with Continuous-time Motion Models}},
    author    = {Zhao, Lingzhe and Wang, Peng and Liu, Peidong},
    year      = {2023},
    note      = {{https://github.com/WU-CVGL/BAD-NeRFs}}
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

- Kudos to the [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) team for their amazing framework:

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
```
