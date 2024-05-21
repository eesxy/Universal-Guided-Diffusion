import sys
import os
from pathlib import Path
import os
import errno
import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddim_with_grad import DDIMSamplerWithGrad
from transformers import AutoFeatureExtractor
from torchvision import transforms, utils
import torch
from torch.utils.data import Subset
from torch.optim import SGD, Adam, AdamW
import PIL
from torch.utils import data
from pathlib import Path
from PIL import Image
from torchvision import transforms, utils
import random
from scripts.helper import OptimizerDetails
import clip
import os
import inspect
import pickle
from imwatermark import WatermarkEncoder
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from ldm.models.diffusion.plms import PLMSSampler
from helper import get_face_text
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.datasets import ImageFolder
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

sys.path.append(str(Path(os.path.abspath(__file__)).parents[1]))

print(sys.path)

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def return_cv2(img, path):
    black = [255, 255, 255]
    img = (img + 1) * 0.5
    utils.save_image(img, path, nrow=1)
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
    return img


def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


class ObjectDetection(nn.Module):
    def __init__(self):
        super().__init__()
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
        self.preprocess = weights.transforms()
        for param in self.model.parameters():
            param.requires_grad = False
        self.categories = weights.meta["categories"]

        print(weights.meta["categories"])

    def forward(self, x):
        self.model.eval()
        inter = self.preprocess((x + 1) * 0.5)
        return self.model(inter)

    def cal_loss(self, x, gt):
        def set_bn_to_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.model.train()
        self.model.backbone.eval()
        self.model.apply(set_bn_to_eval)
        inter = self.preprocess((x + 1) * 0.5)
        loss = self.model(inter, gt)
        return loss['loss_classifier'] + loss['loss_objectness'] + loss['loss_rpn_box_reg']


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image,
                                                       clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, data_aug=False, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        random.shuffle(self.paths)

        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size), resample=PIL.Image.LANCZOS)

        return self.transform(img)


def return_cv2(img, path):
    black = [255, 255, 255]
    img = (img + 1) * 0.5
    utils.save_image(img, path, nrow=1)
    img = cv2.imread(path)
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black)
    return img


def cycle(dl):
    while True:
        for data in dl:
            yield data


import os
import errno


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


class FaceRecognition(nn.Module):
    def __init__(self, fr_crop=False, mtcnn_face=False):
        super().__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        print(self.resnet)
        self.mtcnn = MTCNN(device='cuda')
        self.crop = fr_crop
        self.output_size = 160
        self.mtcnn_face = mtcnn_face

    def extract_face(self, imgs, batch_boxes, mtcnn_face=False):
        image_size = imgs.shape[-1]
        faces = []
        for i in range(imgs.shape[0]):
            img = imgs[i]
            if not mtcnn_face:
                box = [48, 48, 208, 208]
                crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
            elif batch_boxes[i] is not None:
                box = batch_boxes[i][0]
                margin = [
                    self.mtcnn.margin * (box[2] - box[0]) / (self.output_size - self.mtcnn.margin),
                    self.mtcnn.margin * (box[3] - box[1]) / (self.output_size - self.mtcnn.margin),
                ]

                box = [
                    int(max(box[0] - margin[0] / 2, 0)),
                    int(max(box[1] - margin[1] / 2, 0)),
                    int(min(box[2] + margin[0] / 2, image_size)),
                    int(min(box[3] + margin[1] / 2, image_size)), ]
                crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
            else:
                # crop_face = img[None, :, :, :]
                return None

            faces.append(F.interpolate(crop_face, size=self.output_size, mode='bicubic'))
        new_faces = torch.cat(faces)

        return (new_faces - 127.5) / 128.0

    def get_faces(self, x, mtcnn_face=False):
        img = (x + 1.0) * 0.5 * 255.0
        img = img.permute(0, 2, 3, 1)
        with torch.no_grad():
            batch_boxes, batch_probs, batch_points = self.mtcnn.detect(img, landmarks=True)
            # Select faces
            batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.mtcnn.selection_method)

        img = img.permute(0, 3, 1, 2)
        faces = self.extract_face(img, batch_boxes, mtcnn_face)
        return faces

    def forward(self, x, return_faces=False, mtcnn_face=None):
        x = TF.resize(x, (256, 256), interpolation=TF.InterpolationMode.BICUBIC)

        if mtcnn_face is None:
            mtcnn_face = self.mtcnn_face

        faces = self.get_faces(x, mtcnn_face=mtcnn_face)
        if faces is None:
            return faces

        if not self.crop:
            out = self.resnet(x)
        else:
            out = self.resnet(faces)

        if return_faces:
            return out, faces
        else:
            return out

    def cuda(self):
        self.resnet = self.resnet.cuda()
        self.mtcnn = self.mtcnn.cuda()
        return self


def cycle_cat(dl):
    while True:
        for data in dl:
            yield data[0]


def l1_loss(input, target):
    l = torch.abs(input - target).mean(dim=[1])
    return l


def get_optimation_details_od(args):

    guidance_func = ObjectDetection().cuda()
    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = guidance_func

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = None

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff

    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance

    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt_od
    operation.original_guidance = args.optim_original_conditioning

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 5
    operation.folder = args.optim_folder

    return operation


def get_optimation_details_fd(args):
    mtcnn_face = not args.center_face
    print('mtcnn_face')
    print(mtcnn_face)

    guidance_func = FaceRecognition(fr_crop=args.fr_crop, mtcnn_face=mtcnn_face).cuda()
    operation = OptimizerDetails()

    operation.num_steps = args.optim_num_steps
    operation.operation_func = guidance_func

    operation.optimizer = 'Adam'
    operation.lr = args.optim_lr
    operation.loss_func = l1_loss

    operation.max_iters = args.optim_max_iters
    operation.loss_cutoff = args.optim_loss_cutoff

    operation.guidance_3 = args.optim_forward_guidance
    operation.guidance_2 = args.optim_backward_guidance

    operation.optim_guidance_3_wt = args.optim_forward_guidance_wt_fd
    operation.original_guidance = args.optim_original_conditioning

    operation.warm_start = args.optim_warm_start
    operation.print = args.optim_print
    operation.print_every = 20
    operation.folder = args.optim_folder

    return operation


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=500,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.5,
        help=
        "unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument("--optim_lr", default=1e-2, type=float)
    parser.add_argument('--optim_max_iters', type=int, default=1)
    parser.add_argument('--optim_mask_type', type=int, default=1)
    parser.add_argument("--optim_loss_cutoff", default=0.00001, type=float)
    parser.add_argument('--optim_forward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_backward_guidance', action='store_true', default=False)
    parser.add_argument('--optim_original_conditioning', action='store_true', default=False)
    parser.add_argument("--optim_forward_guidance_wt_od", default=100.0, type=float)
    parser.add_argument("--optim_forward_guidance_wt_fd", default=20000.0, type=float)
    parser.add_argument('--optim_do_forward_guidance_norm', action='store_true', default=False)
    parser.add_argument("--optim_tv_loss", default=None, type=float)
    parser.add_argument('--optim_warm_start', action='store_true', default=False)
    parser.add_argument('--optim_print', action='store_true', default=False)
    parser.add_argument('--optim_aug', action='store_true', default=False)
    parser.add_argument('--optim_folder', default='./temp/')
    parser.add_argument("--optim_num_steps", nargs="+", default=[5], type=int)
    parser.add_argument("--optim_mask_fraction", default=0.5, type=float)

    parser.add_argument('--text_type', type=int, default=1)
    parser.add_argument("--text", default=None)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument('--face_folder', default='./data/face_data')

    parser.add_argument('--fr_crop', action='store_true')
    parser.add_argument('--center_face', action='store_true')
    parser.add_argument("--trials", default=20, type=int)
    parser.add_argument("--indexes", nargs="+", default=[0, 1, 2], type=int)

    opt = parser.parse_args()

    results_folder = opt.optim_folder
    create_folder(results_folder)

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.eval()

    sampler = DDIMSamplerWithGrad(model)

    batch_size = opt.n_samples
    assert batch_size == 1

    operation_od = get_optimation_details_od(opt)
    operation_fd = get_optimation_details_fd(opt)

    # face data
    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)])

    batch_size = opt.batch_size

    ds = ImageFolder(root=opt.face_folder, transform=transform)
    dl = data.DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=16,
                         drop_last=True)

    torch.set_grad_enabled(False)

    if opt.text is not None:
        prompt = opt.text
    else:
        prompt = "a headshot of a woman with a dog"

    print(prompt)

    # Setup for object detection

    def gen_box(num_image, anchor_locs, label, sizes):

        objd_cond = []
        for _ in range(num_image):
            boxes = []
            labels = torch.Tensor(label).long()
            for aidx, anchor_loc in enumerate(anchor_locs):
                x, y = anchor_loc
                size = sizes[aidx]
                if isinstance(size, list):
                    x_size = size[0]
                    y_size = size[1]
                else:
                    x_size = size
                    y_size = size
                box = [x - x_size / 2, y - y_size / 2, x + x_size / 2, y + y_size / 2]
                boxes.append(box)
            boxes = torch.Tensor(boxes)
            objd_cond.append({'boxes': boxes.cuda(), 'labels': labels.cuda()})
        return objd_cond

    def draw_box(img, pred):
        labels = [obj_categories[j] for j in pred["labels"].cpu()]
        uint8_image = (img.cpu() * 255).to(torch.uint8)
        box = draw_bounding_boxes(uint8_image, boxes=pred["boxes"].cpu(), labels=labels,
                                  colors="red", width=4)
        box = box.float() / 255.0
        box = box * 2 - 1
        return box

    for index, d in zip(opt.indexes, dl):

        # object detection
        print(f'current bounding box:{index}')
        # Change the bounding box definition here
        if index == 0:
            obj_det_cats = ["dog", "person"]
            test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
            sizes = [180, [200, 400]]
        elif index == 1:
            obj_det_cats = ["person", "dog"]
            test_anchor_locs = [(128.0, 256.0), (384.0, 256.0)]
            sizes = [[200, 400], 180]
        elif index == 2:
            obj_det_cats = ["dog", "person"]
            test_anchor_locs = [(150.0, 384.0), (384.0, 256.0)]
            sizes = [[275, 200], [200, 400]]

        obj_categories = operation_od.operation_func.categories
        category = [obj_categories.index(cc) for cc in obj_det_cats]
        og_img_guide_od = gen_box(opt.n_samples, test_anchor_locs, category, sizes)

        # face detection

        og_img, _ = d
        og_img = og_img.cuda()
        temp = (og_img + 1) * 0.5
        utils.save_image(temp, f'{results_folder}/og_img_{index}.png')

        with torch.no_grad():
            og_img_guide_fd, og_img_mask = operation_fd.operation_func(
                og_img, return_faces=True, mtcnn_face=True)
            utils.save_image((og_img_mask + 1) * 0.5, f'{results_folder}/og_img_cut_{index}.png')

        uc = model.module.get_learned_conditioning(batch_size * [""])
        c = model.module.get_learned_conditioning(batch_size * [prompt])

        for n in trange(opt.trials, desc="Sampling"):

            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
            samples_ddim = sampler.sample_seperate(
                S=opt.ddim_steps,
                conditioning=c,
                batch_size=opt.n_samples,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                unconditional_conditioning=uc,
                eta=opt.ddim_eta,
                operated_image_od=og_img_guide_od,
                operated_image_fd=og_img_guide_fd,
                operation_od=operation_od,
                operation_fd=operation_fd,
            )

            x_samples_ddim = model.module.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            utils.save_image(x_samples_ddim, f'{results_folder}/new_img_{n}.png')
            box_original_output = draw_box(x_samples_ddim[0].detach(), og_img_guide_od[0])
            img_ = return_cv2(box_original_output, f'{results_folder}/box_new_img_{n}.png')


if __name__ == "__main__":
    main()
    """
    python scripts/combine.py --indexes 0 --text "a headshot of a woman with a dog" --scale 1.5 --optim_forward_guidance --optim_num_steps 2 --optim_forward_guidance_wt_od 100 --optim_forward_guidance_wt_fd 20000 --optim_original_conditioning --ddim_steps 500 --face_folder ./data/face_data/ --optim_folder ./test_combine/ --ckpt <Path to stable diffusion model>
    """
