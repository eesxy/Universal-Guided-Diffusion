# Universal Guidance for Diffusion Models

This repository is a improvement of <a href="https://arxiv.org/abs/2302.07121">Universal Guidance for Diffusion Models</a>.

## Stable Diffusion

```bash
cd stable-diffusion-guided
```

The code for stable diffusion is in `stable-diffusion-guided`, and we use the `sd-v1-4.ckpt` checkpoint. Download this model from <a href="https://github.com/CompVis/stable-diffusion">Stable Diffusion</a> and use its location in the scripts.

## Installations

```bash
conda env create -f environment.yaml
conda activate ldm
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install GPUtil
pip install blobfile
pip install facenet-pytorch
```

## Experiments 2: multiple guidances

To reproduce the experiments in section 5, use the following script:

```bash
python scripts/combine.py --indexes 0 --text "a headshot of a woman with a dog" --scale 1.5 --optim_forward_guidance --optim_num_steps 2 --optim_forward_guidance_wt_od 100 --optim_forward_guidance_wt_fd 20000 --optim_original_conditioning --ddim_steps 500 --face_folder ./data/face_data/ --optim_folder ./test_combine/ --ckpt <Path to stable diffusion model>
```
