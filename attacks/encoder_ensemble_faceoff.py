import argparse
import copy
import hashlib
import itertools
import logging
import os
from pathlib import Path

import datasets
import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import torchvision
from transformers import AutoTokenizer, PretrainedConfig
from utils import LatentAttack, agm
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
import numpy as np
import pdb; 
from ip_adapter.resampler import Resampler
import random
from photomaker.photomaker_clip import PhotoMakerIDEncoder
# from PuLID.pulid.pipeline import PuLIDPipeline

import sys

# 替换为您想要加入的目录的相对或绝对路径
directory_path = '/data1/humw/Codes/My-Anti-DreamBooth/attacks/PuLID/eva_clip'
# 获取绝对路径
absolute_path = os.path.abspath(directory_path)
# 将绝对路径添加到sys.path
sys.path.append(absolute_path)

# 替换为您想要加入的目录的相对或绝对路径
directory_path = '/data1/humw/Codes/My-Anti-DreamBooth/attacks/PuLID/'
# 获取绝对路径
absolute_path = os.path.abspath(directory_path)
# 将绝对路径添加到sys.path
sys.path.append(absolute_path)


from PuLID.eva_clip import create_model_and_transforms

logger = get_logger(__name__)

import torchvision.transforms as transforms

def save_image(save_dir, input_dir, perturbed_data):
    os.makedirs(save_dir, exist_ok=True)
    noised_imgs = perturbed_data.detach()
    img_names = [
        str(instance_path).split("/")[-1]
        for instance_path in list(Path(input_dir).iterdir())
    ]
    for img_pixel, img_name in zip(noised_imgs, img_names):
        save_path = os.path.join(save_dir, img_name)
        Image.fromarray(
            img_pixel.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        ).save(save_path)
    print("save images to {}".format(save_dir))
    

class DreamBoothDatasetFromTensor(Dataset):
    """Just like DreamBoothDataset, but take instance_images_tensor instead of path"""

    def __init__(
        self,
        instance_images_tensor,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=512,
        center_crop=False,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_tensor = instance_images_tensor
        self.num_instance_images = len(self.instance_images_tensor)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_tensor[index % self.num_instance_images]
        example["instance_images"] = instance_image
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0", 
        help="gpu id")
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training.")
    parser.add_argument(
        "--eot",
        type=int,
        default=0,
        required=True,
        help="1 use eot, 0 not use eot",
    )
    parser.add_argument(
        "--distance_choice",
        type=str,
        default="mse",
        required=True,
        help="mse or cosine similarity",
    )
    parser.add_argument(
        "--target_image_path",
        default=None,
        help="target image for attacking",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--instance_data_dir_for_adversarial",
        type=str,
        default=None,
        required=True,
        help="A folder containing the images to add adversarial noise",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_adv_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train adversarial noise.",
    )
    parser.add_argument(
        "--pgd_alpha",
        type=float,
        default=1.0 / 255,
        help="The step size for pgd.",
    )
    parser.add_argument(
        "--pgd_eps",
        type=float,
        default=0.05,
        help="The noise budget for pgd.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def load_data(args, data_dir="", size=512, center_crop=True) -> torch.Tensor:
    if args.eot == 0: # 不用eot，加载数据的时候就进行数据变换
        image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        images = [image_transforms(Image.open(i).convert("RGB")) for i in list(Path(data_dir).iterdir())]
        images = torch.stack(images)
    else: # 使用eot，这里加载数据的时候就不进行数据变换
        def image_to_numpy(image):
            return np.array(image).astype(np.uint8)
        # more robust loading to avoid loaing non-image files
        images = [] 
        for i in list(Path(data_dir).iterdir()):
            if not i.suffix in [".jpg", ".png", ".jpeg"]:
                continue
            else:
                images.append(image_to_numpy(Image.open(i).convert("RGB")))
        images = [Image.fromarray(i).resize((size, size), 2) for i in images]
        images = np.stack(images)
        # from B x H x W x C to B x C x H x W
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        assert images.shape[-1] == images.shape[-2]
    return images

def pgd_attack(
    args,
    model_dict,
    perturbed_images: torch.Tensor,
    original_images: torch.Tensor,
    target_images: torch.Tensor,
    num_steps: int,
    trans_224,
    trans_336,
    trans_512,
):
    """Return new perturbed data"""

    weight_dtype = torch.bfloat16
    device = torch.device(args.device)

    perturbed_images = perturbed_images.detach().clone()
    original_images.requires_grad_(False)
    perturbed_images.requires_grad_(True)
    target_images = target_images.requires_grad_(False)

    if args.target_image_path == "none": # 无目标
        # 加入随机扰动
        perturbed_images = (perturbed_images + (torch.rand(*perturbed_images.shape)*2*args.pgd_eps-args.pgd_eps)).to(weight_dtype)
    
    # 获取原始图像和目标图像的编码
    target_embeds_dict = {}
    original_embeds_dict = {}
    for model_type in model_dict.keys():
        if "vae" in model_type:
            tran_original_data_512 = trans_512(original_images)
            tran_target_data_512 = trans_512(target_images)
            original_image_embeds = model_dict[model_type].encode(tran_original_data_512).latent_dist.sample() * model_dict[model_type].config.scaling_factor
            target_image_embeds = model_dict[model_type].encode(tran_target_data_512).latent_dist.sample() * model_dict[model_type].config.scaling_factor
        elif "ipadapter" == model_type:
            tran_original_data_224 = trans_224(original_images)
            tran_target_data_224 = trans_224(target_images)
            original_image_embeds = model_dict[model_type](tran_original_data_224, output_hidden_states=True).hidden_states[-2]
            target_image_embeds = model_dict[model_type](tran_target_data_224, output_hidden_states=True).hidden_states[-2]
        elif "photomaker" == model_type:
            tran_original_data_224 = trans_224(original_images)
            tran_target_data_224 = trans_224(target_images)
            original_image_embeds = model_dict[model_type](tran_original_data_224)
            target_image_embeds = model_dict[model_type](tran_target_data_224)
        elif "pulid" == model_type:
            tran_original_data_336 = trans_336(original_images)
            tran_target_data_336 = trans_336(target_images)
            ori_id_cond_vit, ori_id_vit_hidden = model_dict[model_type](
                tran_original_data_336, return_all_features=False, return_hidden=True, shuffle=False
            )
            original_image_embeds = torch.cat((ori_id_cond_vit, ori_id_vit_hidden), dim=0)
            # 后续如何处理还不明确
            # original_image_embeds = torch.cat((ori_id_cond_vit, ori_id_vit_hidden), dim=0)
            tgt_id_cond_vit, tgt_id_vit_hidden = model_dict[model_type](
                tran_target_data_336, return_all_features=False, return_hidden=True, shuffle=False
            )
            target_image_embeds = torch.cat((tgt_id_cond_vit, tgt_id_vit_hidden), dim=0)
        else:
            raise NotImplementedError
        target_embeds_dict[model_type] = target_image_embeds
        original_embeds_dict[model_type] = original_image_embeds
    
    pgd_loss_list = list() # 保存损失函数字典
    sum_grad = torch.zeros_like(perturbed_images)
    for step in range(num_steps): # 6
        perturbed_images.requires_grad = True
        # 获取对抗图像的编码
        perturbed_embeds_dict = {}
        loss_dict = {}
        grad_dict = {}
        for model_type in model_dict.keys():
            if "vae" in model_type:
                tran_perturbed_data_512 = trans_512(perturbed_images)
                perturbed_image_embeds = model_dict[model_type].encode(tran_perturbed_data_512).latent_dist.sample() * model_dict[model_type].config.scaling_factor
            elif "ipadapter" == model_type:
                tran_perturbed_data_224 = trans_224(perturbed_images)
                perturbed_image_embeds = model_dict[model_type](tran_perturbed_data_224, output_hidden_states=True).hidden_states[-2]
            elif "photomaker" == model_type:
                tran_perturbed_data_224 = trans_224(perturbed_images)
                perturbed_image_embeds = model_dict[model_type](tran_perturbed_data_224)
            elif "pulid" == model_type:
                tran_perturbed_data_336 = trans_336(perturbed_images)
                prt_id_cond_vit, prt_id_vit_hidden = model_dict[model_type](
                    tran_perturbed_data_336, return_all_features=False, return_hidden=True, shuffle=False
                )
                perturbed_image_embeds = torch.cat((prt_id_cond_vit, prt_id_vit_hidden), dim=0)
            else:
                raise NotImplementedError
            perturbed_embeds_dict[model_type] = perturbed_image_embeds
            # 最小化编码器目标损失函数
            if args.distance_choice == "mse":
                loss = F.mse_loss(target_embeds_dict[model_type], perturbed_embeds_dict[model_type], reduction="mean")
            else: # cosine
                loss = - F.cosine_similarity(target_embeds_dict[model_type], perturbed_embeds_dict[model_type], -1).mean()
            loss_dict[model_type] = loss
            grad = torch.autograd.grad(loss, perturbed_images, create_graph=True)[0]
            grad_dict[model_type] = grad
            sum_grad += grad
        print("step: {}, loss_dict: {}".format(step, loss_dict))
        pgd_loss_list.append(loss_dict) # 每轮保存一次loss_dict
        
        adv_perturbed_data = perturbed_images - args.pgd_alpha * sum_grad.sign() # 目标损失最小化，所以是减
        et = torch.clamp(adv_perturbed_data - original_images, min=-args.pgd_eps, max=+args.pgd_eps)
        perturbed_images = torch.clamp(original_images + et, min=torch.min(original_images), max=torch.max(original_images)).detach().clone()
    return perturbed_images, pgd_loss_list

def main(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if args.mixed_precision == "fp32":
        torch_dtype = torch.float32
    elif args.mixed_precision == "fp16":
        torch_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    model_type_list = ["vae14", "vae15", "vae21", "ipadapter", "photomaker", "pulid",]
    model_dict = {}
    for model_type in model_type_list:
        if "vae14" == model_type: # 加载VAE模型
            model = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path="/data1/humw/Pretrains/stable-diffusion-v1-4", 
                subfolder="vae", 
                revision=args.revision
            ).to(dtype=torch_dtype).eval().requires_grad_(False)
        elif "vae15" == model_type: # 加载VAE模型
            model = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path="/data1/humw/Pretrains/stable-diffusion-v1-5", 
                subfolder="vae", 
                revision=args.revision
            ).to(dtype=torch_dtype).eval().requires_grad_(False)
        elif "vae21" == model_type: # 加载VAE模型
            model = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path="/data1/humw/Pretrains/stable-diffusion-2-1-base", 
                subfolder="vae", 
                revision=args.revision
            ).to(dtype=torch_dtype).eval().requires_grad_(False)
        elif model_type == "ipadapter":
            ipadapter_path = "/data1/humw/Pretrains/IP-Adapter/models/image_encoder"
            model = CLIPVisionModelWithProjection.from_pretrained(ipadapter_path).to(dtype=torch_dtype).eval().requires_grad_(False)
        elif model_type == "photomaker":
            model = PhotoMakerIDEncoder()
            state_dict = torch.load("/data1/humw/Pretrains/photomaker-v1.bin", map_location="cpu")
            model.load_state_dict(state_dict['id_encoder'], strict=True)
            model.to(dtype=torch_dtype).eval().requires_grad_(False)
        elif model_type == "pulid":
            # clip-vit backbone
            model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
            model = model.visual
            model.to(dtype=torch_dtype).eval().requires_grad_(False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        model_dict[model_type] = model
    # pdb.set_trace()

    perturbed_data = load_data(
        args,
        data_dir=args.instance_data_dir_for_adversarial,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    target_data = load_data(
        args,
        data_dir=args.target_image_path,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    original_data = perturbed_data.clone()
    original_data.requires_grad_(False)
    
    resample_interpolation = transforms.InterpolationMode.BILINEAR
    if args.eot == 0: # 不用eot，不过ipadapter因为尺寸不一致不得不用eot
        trans_224 = [
            transforms.Resize(224, interpolation=resample_interpolation),
            transforms.CenterCrop(224) if args.center_crop else transforms.RandomCrop(224),
        ]
        trans_224 = transforms.Compose(trans_224)
        
        trans_336 = [
            transforms.Resize(336, interpolation=resample_interpolation),
            transforms.CenterCrop(336) if args.center_crop else transforms.RandomCrop(336),
        ]
        trans_336 = transforms.Compose(trans_336)
        
        trans_512 = None # 默认512所以不再对512尺寸模型输入进行处理
    else: # 使用eot
        train_aug_224 = [
            transforms.Resize(224, interpolation=resample_interpolation),
            transforms.CenterCrop(224) if args.center_crop else transforms.RandomCrop(224),
        ]

        tensorize_and_normalize = [
            transforms.Normalize([0.5*255]*3,[0.5*255]*3),
        ]
        trans_224 = train_aug_224 + tensorize_and_normalize
        trans_224 = transforms.Compose(trans_224)
        print("all_trans:{}".format(trans_224))
        
        train_aug_336 = [
            transforms.Resize(336, interpolation=resample_interpolation),
            transforms.CenterCrop(336) if args.center_crop else transforms.RandomCrop(336),
        ]

        trans_336 = train_aug_336 + tensorize_and_normalize
        trans_336 = transforms.Compose(trans_336)
        print("all_trans:{}".format(trans_336))
        
        train_aug_512 = [
            transforms.Resize(512, interpolation=resample_interpolation),
            transforms.CenterCrop(512) if args.center_crop else transforms.RandomCrop(512),
        ]
        trans_512 = train_aug_512 + tensorize_and_normalize
        trans_512 = transforms.Compose(trans_512)
        print("all_trans:{}".format(trans_512))
        
        args.pgd_eps = 16.0
        args.pgd_alpha = 16/10 # 默认是阈值的1/10
        
    pgd_loss_list = []
    for i in range(args.max_train_steps):
        perturbed_data, tmp_pgd_loss_list  = pgd_attack(
            args,
            model_dict=model_dict,
            data_tensor=perturbed_data,
            original_images=original_data,
            target_tensor=target_data,
            num_steps=args.max_adv_train_steps,
            trans_224=trans_224,
            trans_336=trans_336,
            trans_512=trans_512,
        )
        pgd_loss_list.extend(tmp_pgd_loss_list)

    save_folder = args.output_dir
    os.makedirs(save_folder, exist_ok=True)
    noised_imgs = perturbed_data.detach()
    img_names = [
        str(instance_path).split("/")[-1]
        for instance_path in list(Path(args.instance_data_dir_for_adversarial).iterdir())
    ]
    if args.eot == 0:
        for img_pixel, img_name in zip(noised_imgs, img_names):
            save_path = os.path.join(save_folder, img_name)
            Image.fromarray(
                (img_pixel * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            ).save(save_path)
    else:
        save_image(save_folder, args.instance_data_dir_for_adversarial, noised_imgs)
    print(f"Saved noise images to {save_folder}")
    
    # 保存PGD攻击损失列表
    person_id = args.output_dir.split('/')[-1]
    exp_name = args.output_dir.split('/')[-2] # 倒数第一是person_id，倒数第二是exp_name
    config_scripts_logs_path = "/data1/humw/Codes/My-Anti-DreamBooth/outputs/config_scripts_logs/" + exp_name
    os.makedirs(config_scripts_logs_path, exist_ok=True)
    with open(f"{config_scripts_logs_path}/{person_id}_pgd_loss_list.txt", "w") as f:
        # f.write(person_id + '\n')
        f.write(str(pgd_loss_list) + "\n")
        for index, loss_dict in enumerate(pgd_loss_list):
            f.write("index: " + str(index) + ", " + str(loss_dict) + "\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    