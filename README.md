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

## Experiments 1: universal guidance with more liberalization
Execute the following script on a single NVIDIA RTX 4090, where `--optim_original_conditioning` is the basic scheme for universal guidance generation, `--optim_mix_conditioning` is scheme 1 for liberalization guidance, and `--optim_mix_conditioning2` is scheme 2 for liberalization guidance. The following script explores the guidance effects of each guidance scheme on three downstream tasks: object detection, facial recognition, and style transfer, using stable diffusion model v1-4 as the basic model.
```
#!/usr/bin/env bash
echo "开始测试......"
python scripts/style_transfer.py --indexes 0 --text "A colorful photo of a eiffel tower" --scale 3.0 --optim_forward_guidance --optim_num_steps 6 --optim_forward_guidance_wt 6 --optim_original_conditioning --ddim_steps 500 --optim_folder ./test_style/text_type_1/ --ckpt sd-v1-4.ckpt --trials 3
wait
python scripts/style_transfer.py --indexes 0 --text "A colorful photo of a eiffel tower" --scale 3.0 --optim_forward_guidance --optim_num_steps 6 --optim_forward_guidance_wt 6 --optim_mix_conditioning --ddim_steps 500 --optim_folder ./test_style/text_type_2/ --ckpt sd-v1-4.ckpt --trials 3
wait
python scripts/style_transfer.py --indexes 0 --text "A colorful photo of a eiffel tower" --scale 3.0 --optim_forward_guidance --optim_num_steps 6 --optim_forward_guidance_wt 6 --optim_mix_conditioning2 --ddim_steps 500 --optim_folder ./test_style/text_type_3/ --ckpt sd-v1-4.ckpt --trials 3
wait
python scripts/object_detection.py --indexes 0 --text "a headshot of a woman with a dog" --scale 1.5 --optim_forward_guidance --optim_num_steps 5 --optim_forward_guidance_wt 100 --optim_original_conditioning --ddim_steps 250 --optim_folder ./test_od/text_type_1/ --ckpt sd-v1-4.ckpt --trials 3
wait
python scripts/object_detection.py --indexes 0 --text "a headshot of a woman with a dog" --scale 1.5 --optim_forward_guidance --optim_num_steps 5 --optim_forward_guidance_wt 100 --optim_mix_conditioning --ddim_steps 250 --optim_folder ./test_od/text_type_2/ --ckpt sd-v1-4.ckpt --trials 3
wait
python scripts/object_detection.py --indexes 0 --text "a headshot of a woman with a dog" --scale 1.5 --optim_forward_guidance --optim_num_steps 5 --optim_forward_guidance_wt 100 --optim_mix_conditioning2 --ddim_steps 250 --optim_folder ./test_od/text_type_3/ --ckpt sd-v1-4.ckpt --trials 3
wait
python scripts/face_detection.py --indexes 0 --text "Headshot of a person with blonde hair with space background" --optim_forward_guidance --fr_crop --optim_num_steps 2 --optim_forward_guidance_wt 20000 --optim_original_conditioning --ddim_steps 500 --optim_folder ./test_face/text_type_1/ --ckpt sd-v1-4.ckpt --trials 3
wait
python scripts/face_detection.py --indexes 0 --text "Headshot of a person with blonde hair with space background" --optim_forward_guidance --fr_crop --optim_num_steps 2 --optim_forward_guidance_wt 20000 --optim_mix_conditioning --ddim_steps 500 --optim_folder ./test_face/text_type_2/ --ckpt sd-v1-4.ckpt --trials 3
wait
python scripts/face_detection.py --indexes 0 --text "Headshot of a person with blonde hair with space background" --optim_forward_guidance --fr_crop --optim_num_steps 2 --optim_forward_guidance_wt 20000 --optim_mix_conditioning2 --ddim_steps 500 --optim_folder ./test_face/text_type_3/ --ckpt sd-v1-4.ckpt --trials 3
echo "结束测试......"
```

## Experiments 2: multiple guidances

To reproduce the experiments in section 5, use the following script:

```bash
python scripts/combine.py --indexes 0 --text "a headshot of a woman with a dog" --scale 1.5 --optim_forward_guidance --optim_num_steps 2 --optim_forward_guidance_wt_od 100 --optim_forward_guidance_wt_fd 20000 --optim_original_conditioning --ddim_steps 500 --face_folder ./data/face_data/ --optim_folder ./test_combine/ --ckpt <Path to stable diffusion model>
```
