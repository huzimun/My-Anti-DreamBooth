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
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
import numpy as np
import pdb; 
from ip_adapter.resampler import Resampler
import random
# from photomaker.photomaker_clip import PhotoMakerIDEncoder
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

import torch
import torch.nn as nn
from transformers.models.clip.configuration_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection

VISION_CONFIG_DICT = {
    "hidden_size": 1024,
    "intermediate_size": 4096,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768
}

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x


class FuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, prompt_embeds, id_embeds):
        stacked_id_embeds = torch.cat([prompt_embeds, id_embeds], dim=-1)
        stacked_id_embeds = self.mlp1(stacked_id_embeds) + prompt_embeds
        stacked_id_embeds = self.mlp2(stacked_id_embeds)
        stacked_id_embeds = self.layer_norm(stacked_id_embeds)
        return stacked_id_embeds

    def forward(
        self,
        prompt_embeds,
        id_embeds,
        class_tokens_mask,
    ) -> torch.Tensor:
        # id_embeds shape: [b, max_num_inputs, 1, 2048]
        id_embeds = id_embeds.to(prompt_embeds.dtype)
        num_inputs = class_tokens_mask.sum().unsqueeze(0) # TODO: check for training case
        batch_size, max_num_inputs = id_embeds.shape[:2]
        # seq_length: 77
        seq_length = prompt_embeds.shape[1]
        # flat_id_embeds shape: [b*max_num_inputs, 1, 2048]
        flat_id_embeds = id_embeds.view(
            -1, id_embeds.shape[-2], id_embeds.shape[-1]
        )
        # valid_id_mask [b*max_num_inputs]
        valid_id_mask = (
            torch.arange(max_num_inputs, device=flat_id_embeds.device)[None, :]
            < num_inputs[:, None]
        )
        valid_id_embeds = flat_id_embeds[valid_id_mask.flatten()]

        prompt_embeds = prompt_embeds.view(-1, prompt_embeds.shape[-1])
        class_tokens_mask = class_tokens_mask.view(-1)
        valid_id_embeds = valid_id_embeds.view(-1, valid_id_embeds.shape[-1])
        # slice out the image token embeddings
        image_token_embeds = prompt_embeds[class_tokens_mask]
        stacked_id_embeds = self.fuse_fn(image_token_embeds, valid_id_embeds)
        assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"
        prompt_embeds.masked_scatter_(class_tokens_mask[:, None], stacked_id_embeds.to(prompt_embeds.dtype))
        updated_prompt_embeds = prompt_embeds.view(batch_size, seq_length, -1)
        return updated_prompt_embeds

# modified ID Encoder of PhotoMaker, batch size dimension is ignored and default to 1, the fuse module is ignored
class PhotoMakerIDEncoder(CLIPVisionModelWithProjection):
    def __init__(self):
        super().__init__(CLIPVisionConfig(**VISION_CONFIG_DICT))
        self.visual_projection_2 = nn.Linear(1024, 1280, bias=False)
        self.fuse_module = FuseModule(2048)

    def forward(self, id_pixel_values):
        num_inputs, c, h, w = id_pixel_values.shape
        id_pixel_values = id_pixel_values.view(num_inputs, c, h, w)

        shared_id_embeds = self.vision_model(id_pixel_values)[1]
        id_embeds = self.visual_projection(shared_id_embeds)
        id_embeds_2 = self.visual_projection_2(shared_id_embeds)

        id_embeds = id_embeds.view(num_inputs, 1, -1)
        id_embeds_2 = id_embeds_2.view(num_inputs, 1, -1)

        id_embeds = torch.cat((id_embeds, id_embeds_2), dim=-1)

        return id_embeds
    
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
        "--mode", 
        type=str, 
        default="idprotector", 
        help="idprotector use projection image embeds for ip-adapter")
    parser.add_argument(
        "--norm", 
        type=int, 
        default=0, 
        help="use normalization for agm")
    parser.add_argument(
        "--agm", 
        type=int, 
        default=0, 
        help="use agm or not")
    parser.add_argument(
        "--model_types", 
        type=str, 
        default="vae15", 
        help="model types string split with ;")
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
        "--target",
        type=str,
        default="yingbu",
        required=True,
        help="yingbu, mist, non-target",
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

def vae_gen(perturbed_images, trans_512, vae, device, weight_dtype):
    perturbed_images.requires_grad = False
    if trans_512 is not None:
        trans_perturbed_images = trans_512(perturbed_images).to(device, dtype=weight_dtype)
        latents = vae.encode(trans_perturbed_images).latent_dist.sample()
    else:
        latents = vae.encode(perturbed_images.to(device, dtype=weight_dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor # torch.Size([4, 4, 64, 64])
    return latents

def ipadapter_gen(perturbed_images, trans_224, ipadapter, device, weight_dtype):
    perturbed_images.requires_grad = False
    trans_perturbed_images = trans_224(perturbed_images).to(device, dtype=weight_dtype)
    ipadapter_perturbed_tensor = ipadapter(trans_perturbed_images, output_hidden_states=True).hidden_states[-2]
    return ipadapter_perturbed_tensor

def photomaker_gen(perturbed_images, trans_224, photomaker, device, weight_dtype):
    perturbed_images.requires_grad = False
    trans_perturbed_data_224 = trans_224(perturbed_images).to(device, dtype=weight_dtype)
    photomaker_perturbed_embeds = photomaker(trans_perturbed_data_224)
    return photomaker_perturbed_embeds

def get_adv_example(args, ori_data, adv_data, grad, attack_step=None):
    """
    :param ori_data: original image
    :param adv_data: adversarial image in the last iteration
    :param grad: gradient in this iteration
    :return: adversarial example in this iteration
    """
    if attack_step is None:
        adv_example = adv_data.detach() + grad.sign() * args.pgd_alpha
    else:
        adv_example = adv_data.detach() + grad.sign() * attack_step
    delta = torch.clamp(adv_example - ori_data.detach(), -args.pgd_eps, args.pgd_eps)
    return torch.clamp(ori_data.detach() + delta, max=torch.max(ori_data), min=torch.min(ori_data))

def agm(args, ori_data, cur_adv, grads_dict, target_embeds_dict, original_embeds_dict, 
        beta, models_dict, trans_224, trans_336, trans_512, device, weight_dtype):
    """
    Adaptive gradient modulation
    :param ori_data: natural images
    :param cur_adv: adv examples in last iteration
    :param grad: gradient in this iteration
    :param label: ground truth
    :return: coefficient of each model
    """
    model_types = list()
    for model_type in models_dict.keys():
        model_types.append(model_type)
    # generate adversarial example
    ori_data = ori_data.detach().clone()
    ori_data.requires_grad = False
    cur_adv = cur_adv.detach().clone()
    cur_adv.requires_grad = False
    adv_exp_dict = {}
    loss_self_dict = {}
    for model_type in model_types:
        adv_exp = get_adv_example(args, ori_data=ori_data, adv_data=cur_adv, grad=grads_dict[model_type])
        adv_exp_dict[model_type] = adv_exp
        if "vae" in model_type:
            latents = vae_gen(adv_exp_dict[model_type], trans_512, models_dict[model_type], device, weight_dtype)
            if args.target == "non-target": # 无目标攻击
                if args.distance_choice == "cosine":
                    loss_vae = F.cosine_similarity(latents.to(weight_dtype), original_embeds_dict[model_type].to(weight_dtype), -1).mean().item()
                else: # mse or mix
                    loss_vae = - F.mse_loss(latents.to(weight_dtype), original_embeds_dict[model_type].to(weight_dtype), reduction="mean").item()
            else: # 目标攻击
                if args.distance_choice == "cosine":
                    loss_vae = - F.cosine_similarity(latents.to(weight_dtype), target_embeds_dict[model_type].to(weight_dtype), -1).mean().item()
                else: # mse or mix
                    loss_vae = F.mse_loss(latents.to(weight_dtype), target_embeds_dict[model_type].to(weight_dtype), reduction="mean").item()
            loss_self_dict[model_type] = loss_vae
            latents.to("cpu")
            del latents
        elif model_type == "ipadapter":
            ipadapter_perturbed_tensor = ipadapter_gen(adv_exp_dict[model_type], trans_224, models_dict[model_type], device, weight_dtype)
            if args.target == "non-target": # 无目标攻击
                if args.distance_choice == "cosine" or args.distance_choice == "mix":
                    loss_ipadapter = F.cosine_similarity(ipadapter_perturbed_tensor.to(weight_dtype), original_embeds_dict[model_type].to(weight_dtype), -1).mean().item() # -min=max
                else:
                    loss_ipadapter = - F.mse_loss(ipadapter_perturbed_tensor.to(weight_dtype), original_embeds_dict[model_type].to(weight_dtype)).item()
            else: # 目标攻击
                if args.distance_choice == "cosine" or args.distance_choice == "mix":
                    loss_ipadapter = - F.cosine_similarity(ipadapter_perturbed_tensor.to(weight_dtype), target_embeds_dict[model_type].to(weight_dtype), -1).mean().item() # -min=max
                else:
                    loss_ipadapter = F.mse_loss(ipadapter_perturbed_tensor.to(weight_dtype), target_embeds_dict[model_type].to(weight_dtype)).item()
            loss_self_dict[model_type] = loss_ipadapter
            ipadapter_perturbed_tensor.to("cpu")
            del ipadapter_perturbed_tensor
        elif model_type == "photomaker":
            photomaker_perturbed_tensor = photomaker_gen(adv_exp_dict[model_type], trans_224, models_dict[model_type], device, weight_dtype)
            if args.target == "non-target": # 无目标攻击
                if args.distance_choice == "cosine" or args.distance_choice == "mix":
                    loss_photomaker = F.cosine_similarity(photomaker_perturbed_tensor.to(weight_dtype), original_embeds_dict[model_type].to(weight_dtype), -1).mean().item() # -min=max
                else:
                    loss_photomaker = - F.mse_loss(photomaker_perturbed_tensor.to(weight_dtype), original_embeds_dict[model_type].to(weight_dtype)).item()
            else: # 目标攻击
                if args.distance_choice == "cosine" or args.distance_choice == "mix":
                    loss_photomaker = - F.cosine_similarity(photomaker_perturbed_tensor.to(weight_dtype), target_embeds_dict[model_type].to(weight_dtype), -1).mean().item() # -min=max
                else:
                    loss_photomaker = F.mse_loss(photomaker_perturbed_tensor.to(weight_dtype), target_embeds_dict[model_type].to(weight_dtype)).item()
            loss_self_dict[model_type] = loss_photomaker
            photomaker_perturbed_tensor.to("cpu")
            del photomaker_perturbed_tensor
        else:
            raise ValueError('Wrong model index')
    # pdb.set_trace()
    w = torch.zeros(size=(len(model_types),), device=models_dict[model_types[0]].device)
    
    # import pdb; pdb.set_trace()
    for j, model_type_j in enumerate(model_types):
        for i, model_type_i in enumerate(model_types):
            if model_type_i == model_type_j:
                continue
            elif "vae" in model_type_i: # vae
                latents = vae_gen(adv_exp_dict[model_type_j], trans_512, models_dict[model_type_i], device, weight_dtype) # 第j个对抗样本在模型i上的输出
                if args.target == "non-target": # 无目标
                    if args.distance_choice == "cosine":
                        loss = F.cosine_similarity(latents.to(weight_dtype), original_embeds_dict[model_type_i].to(weight_dtype), -1).mean().item()
                    else:
                        loss = - F.mse_loss(latents.to(weight_dtype), original_embeds_dict[model_type_i].to(weight_dtype), reduction="mean").item()
                else:
                    if args.distance_choice == "cosine":
                        loss = - F.cosine_similarity(latents.to(weight_dtype), target_embeds_dict[model_type_i].to(weight_dtype), -1).mean().item()
                    else:
                        loss = F.mse_loss(latents.to(weight_dtype), target_embeds_dict[model_type_i].to(weight_dtype), reduction="mean").item()
                latents.to("cpu")
                del latents
            elif model_type_i == "ipadapter": # ipadapter
                ipadapter_perturbed_tensor = ipadapter_gen(adv_exp_dict[model_type_j], trans_224, models_dict[model_type_i], device, weight_dtype)
                if args.target == "non-target": # 无目标
                    if args.distance_choice == "cosine" or args.distance_choice == "mix":
                        loss = F.cosine_similarity(ipadapter_perturbed_tensor.to(weight_dtype), original_embeds_dict[model_type_i].to(weight_dtype), -1).mean().item()
                    else:
                        loss = - F.mse_loss(ipadapter_perturbed_tensor.to(weight_dtype), original_embeds_dict[model_type_i].to(weight_dtype)).item()
                else:
                    if args.distance_choice == "cosine" or args.distance_choice == "mix":
                        loss = - F.cosine_similarity(ipadapter_perturbed_tensor.to(weight_dtype), target_embeds_dict[model_type_i].to(weight_dtype), -1).mean().item()
                    else:
                        loss = F.mse_loss(ipadapter_perturbed_tensor.to(weight_dtype), target_embeds_dict[model_type_i].to(weight_dtype)).item()
                ipadapter_perturbed_tensor.to("cpu")
                del ipadapter_perturbed_tensor
            elif model_type_i == "photomaker": # photomaker
                photomaker_perturbed_tensor = photomaker_gen(adv_exp_dict[model_type_j], trans_224, models_dict[model_type_i], device, weight_dtype)
                if args.target == "non-target": # 无目标
                    if args.distance_choice == "cosine" or args.distance_choice == "mix":
                        loss = F.cosine_similarity(photomaker_perturbed_tensor.to(weight_dtype), original_embeds_dict[model_type_i].to(weight_dtype), -1).mean().item()
                    else:
                        loss = - F.mse_loss(photomaker_perturbed_tensor.to(weight_dtype), original_embeds_dict[model_type_i].to(weight_dtype)).item()
                else:
                    if args.distance_choice == "cosine" or args.distance_choice == "mix":
                        loss = - F.cosine_similarity(photomaker_perturbed_tensor.to(weight_dtype), target_embeds_dict[model_type_i].to(weight_dtype), -1).mean().item()
                    else:
                        loss = F.mse_loss(photomaker_perturbed_tensor.to(weight_dtype), target_embeds_dict[model_type_i].to(weight_dtype)).item()
                photomaker_perturbed_tensor.to("cpu")
                del photomaker_perturbed_tensor
            else:
                raise Exception('Wrong model index')
            w[j] += loss / loss_self_dict[model_type_i] * beta # 模型j的对抗样本扰动模型i的损失
    # pdb.set_trace()
    # 发现softmax后概率并不是严格按照数值大小的比例计算的，这样会导致，差不多大小的数值
    if args.norm == 0:
        print("w before softmax: {}".format(w))
        w = torch.softmax(w, dim=0) # 根据模拟的下一步损失确定当前的梯度权重
        print("w after softmax: {}".format(w))
        for i in range(len(w)):
            w[i] = w[i] * len(w)
        print("w after scaling: {}".format(w))
    else:
        if len(w) > 1:
            print("w before normalization: {}".format(w))
            sum_w = torch.sum(w) # 累加求和
            new_w = torch.zeros(size=(len(model_types),), device=models_dict[model_types[0]].device)
            for i in range(len(w)):
                new_w[i] = (w[i] / sum_w) * len(w) # 严格按照比例放缩
            w = new_w
            print("w after normalization: {}".format(w))
        else:
            w = torch.ones(size=(len(model_types),), device=models_dict[model_types[0]].device)
    del ori_data
    del cur_adv
    return w

def pgd_attack(
    args,
    torch_dtype,
    model_dict,
    perturbed_images: torch.Tensor,
    original_images: torch.Tensor,
    target_images: torch.Tensor,
    num_steps: int,
    trans_224,
    trans_336,
    trans_512,
    image_proj_model,
):
    """Return new perturbed data"""
    device = torch.device(args.device)
    perturbed_images = perturbed_images.detach().clone().to(dtype=torch_dtype).to(device)
    # perturbed_images.requires_grad_(True)
    original_images = original_images.requires_grad_(False).to(dtype=torch_dtype).to(device)

    if args.target == "non-target": # 无目标
        # 加入随机扰动
        perturbed_images = (perturbed_images + (torch.rand(*perturbed_images.shape)*2*args.pgd_eps-args.pgd_eps).to(torch_dtype).to(device))
    else:
        target_images = target_images.requires_grad_(False).to(dtype=torch_dtype).to(device)
    
    model_types = list()
    # 遍历模型字典，将每个模型放到gpu上
    for model_type in model_dict.keys():
        model_dict[model_type].eval()
        model_dict[model_type].to(device)
        model_types.append(model_type)
        
    # 获取原始图像和目标图像的编码
    target_embeds_dict = {}
    original_embeds_dict = {}
    for model_type in model_dict.keys():
        if "vae" in model_type:
            tran_original_data_512 = trans_512(original_images).to(dtype=torch_dtype)
            original_image_embeds = model_dict[model_type].encode(tran_original_data_512).latent_dist.sample() * model_dict[model_type].config.scaling_factor
            if args.target != "non-target":
                tran_target_data_512 = trans_512(target_images).to(dtype=torch_dtype)
                target_image_embeds = model_dict[model_type].encode(tran_target_data_512).latent_dist.sample() * model_dict[model_type].config.scaling_factor
        elif "ipadapter" == model_type:
            tran_original_data_224 = trans_224(original_images).to(dtype=torch_dtype)
            original_image_embeds = model_dict[model_type](tran_original_data_224, output_hidden_states=True).hidden_states[-2]
            if image_proj_model is not None:
                proj_original_image_embeds = image_proj_model(original_image_embeds)
                original_image_embeds = proj_original_image_embeds
            if args.target != "non-target":
                tran_target_data_224 = trans_224(target_images).to(dtype=torch_dtype)
                target_image_embeds = model_dict[model_type](tran_target_data_224, output_hidden_states=True).hidden_states[-2]
                if image_proj_model is not None:
                    proj_target_image_embeds = image_proj_model(target_image_embeds)
                    target_image_embeds = proj_target_image_embeds
        elif "photomaker" == model_type:
            tran_original_data_224 = trans_224(original_images).to(dtype=torch_dtype)
            original_image_embeds = model_dict[model_type](tran_original_data_224)
            if args.target != "non-target":
                tran_target_data_224 = trans_224(target_images).to(dtype=torch_dtype)
                target_image_embeds = model_dict[model_type](tran_target_data_224)
        elif "pulid" == model_type:
            tran_original_data_336 = trans_336(original_images).to(dtype=torch_dtype)
            #  ori_id_vit_hidden是一个list，5个torch.Size([4, 577, 1024])的特征
            _, ori_id_vit_hidden = model_dict[model_type]( # ori_id_cond_vit.shape=torch.Size([4, 768])
                tran_original_data_336, return_all_features=False, return_hidden=True, shuffle=False
            )
            # pulid同时利用了ori_id_cond_vit, ori_id_vit_hidden，还利用了人脸检测器的特征编码
            # 我没有利用人脸特征提取器，因为pulid和anti-dreambooth的环境冲突
            # 只利用clip的隐层特征，5层，直接求和
            stacked_hidden = torch.stack(ori_id_vit_hidden, dim=0)
            # 沿着堆叠后的新维度（即第一个维度）求和，得到尺寸为 [4, 577, 1024] 的张量
            original_image_embeds = stacked_hidden.sum(dim=0)
            
            if args.target != "non-target":
                tran_target_data_336 = trans_336(target_images).to(dtype=torch_dtype)
                _, tgt_id_vit_hidden = model_dict[model_type](
                    tran_target_data_336, return_all_features=False, return_hidden=True, shuffle=False
                )
                stacked_hidden = torch.stack(tgt_id_vit_hidden, dim=0)
                # 沿着堆叠后的新维度（即第一个维度）求和，得到尺寸为 [4, 577, 1024] 的张量
                target_image_embeds = stacked_hidden.sum(dim=0)
        else:
            raise NotImplementedError
        if args.target != "non-target":
            target_embeds_dict[model_type] = target_image_embeds
        original_embeds_dict[model_type] = original_image_embeds
    
    pgd_loss_list = list() # 保存损失函数字典
    for step in range(num_steps): # 6
        perturbed_images.requires_grad = True
        # 获取对抗图像的编码
        perturbed_embeds_dict = {}
        loss_dict = {}
        grad_dict = {}
        for model_type in model_dict.keys():
            if "vae" in model_type:
                tran_perturbed_data_512 = trans_512(perturbed_images).to(dtype=torch_dtype)
                perturbed_image_embeds = model_dict[model_type].encode(tran_perturbed_data_512).latent_dist.sample() * model_dict[model_type].config.scaling_factor
            elif "ipadapter" == model_type:
                tran_perturbed_data_224 = trans_224(perturbed_images).to(dtype=torch_dtype)
                perturbed_image_embeds = model_dict[model_type](tran_perturbed_data_224, output_hidden_states=True).hidden_states[-2]
                if image_proj_model is not None:
                    proj_perturbed_image_embeds = image_proj_model(perturbed_image_embeds)
                    perturbed_image_embeds = proj_perturbed_image_embeds
            elif "photomaker" == model_type:
                tran_perturbed_data_224 = trans_224(perturbed_images).to(dtype=torch_dtype)
                perturbed_image_embeds = model_dict[model_type](tran_perturbed_data_224)
            elif "pulid" == model_type:
                tran_perturbed_data_336 = trans_336(perturbed_images).to(dtype=torch_dtype)
                _, prt_id_vit_hidden = model_dict[model_type](
                    tran_perturbed_data_336, return_all_features=False, return_hidden=True, shuffle=False
                )
                stacked_hidden = torch.stack(prt_id_vit_hidden, dim=0)
                # 沿着堆叠后的新维度（即第一个维度）求和，得到尺寸为 [4, 577, 1024] 的张量
                perturbed_image_embeds = stacked_hidden.sum(dim=0)
            else:
                raise NotImplementedError
            perturbed_embeds_dict[model_type] = perturbed_image_embeds
            model_dict[model_type].zero_grad()
            if args.target == "non-target":
                if args.distance_choice == "mse": # 和原始编码MSE距离越大越好，取负，越小越好
                    loss = - F.mse_loss(original_embeds_dict[model_type], perturbed_embeds_dict[model_type], reduction="mean")
                elif args.distance_choice == "cosine": # 和原始编码余弦相似度越小越好
                    loss = F.cosine_similarity(original_embeds_dict[model_type], perturbed_embeds_dict[model_type], -1).mean()
                else: # mix
                    if "vae" in model_type:
                        loss = - F.mse_loss(original_embeds_dict[model_type], perturbed_embeds_dict[model_type], reduction="mean")
                    else:
                        loss = F.cosine_similarity(original_embeds_dict[model_type], perturbed_embeds_dict[model_type], -1).mean()
            else: # 最小化编码器目标损失函数
                if args.distance_choice == "mse":
                    loss = F.mse_loss(target_embeds_dict[model_type], perturbed_embeds_dict[model_type], reduction="mean")
                elif args.distance_choice == "cosine":
                    loss = - F.cosine_similarity(target_embeds_dict[model_type], perturbed_embeds_dict[model_type], -1).mean()
                else: # mix
                    if "vae" in model_type:
                        loss = F.mse_loss(target_embeds_dict[model_type], perturbed_embeds_dict[model_type], reduction="mean")
                    else:
                        loss = - F.cosine_similarity(target_embeds_dict[model_type], perturbed_embeds_dict[model_type], -1).mean()
            loss_dict[model_type] = loss
            grad = torch.autograd.grad(loss, perturbed_images, retain_graph=True, create_graph=False)[0]
            grad_dict[model_type] = grad
        weighted_loss = 0.0
        sum_loss = 0.0
        if args.agm == 1:
            # pdb.set_trace()
            alphas = agm(args, ori_data=original_images, cur_adv=perturbed_images, grads_dict=grad_dict,
                        target_embeds_dict=target_embeds_dict, original_embeds_dict=original_embeds_dict, 
                        beta=10, models_dict=model_dict, trans_224=trans_224, trans_336=trans_336, trans_512=trans_512,
                        device=device, weight_dtype=torch_dtype)
            print("alpha: {}".format(alphas))
            for tmp_idx, model_type in enumerate(model_types):
                weighted_loss += alphas[tmp_idx] * loss_dict[model_type]
                sum_loss += loss_dict[model_type]
        elif args.agm == 2:
            alphas = [10, 1, 1]
            for tmp_idx, model_type in enumerate(model_types):
                sum_loss += alphas[tmp_idx] * loss_dict[model_type]
            weighted_loss = sum_loss
        else: # 取平均
            for tmp_idx, model_type in enumerate(model_types):
                sum_loss += loss_dict[model_type]
            weighted_loss = sum_loss
        loss_dict["sum_loss"] = sum_loss
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key].item()
        print("Step: {}, loss_dict: {}".format(step, loss_dict))
        pgd_loss_list.append(loss_dict) # 每轮保存一次loss_dict
        # import pdb; pdb.set_trace()
        weighted_grad = torch.autograd.grad(weighted_loss, perturbed_images)[0]
        adv_perturbed_data = perturbed_images - args.pgd_alpha * weighted_grad.sign() # 目标损失最小化，所以是减
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
    model_type_list = args.model_types.split("-")
    model_dict = {}
    print("model_type_list: ", model_type_list)
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
            if args.mode == "idprotector":
                # 映射层参数参考IPAdapterPlusXL中的默认配置
                image_proj_model = Resampler(
                    dim=1280,
                    depth=4,
                    dim_head=64,
                    heads=20,
                    num_queries= 16, # self.num_tokens,
                    embedding_dim= 1280, # self.image_encoder.config.hidden_size,
                    output_dim=2048,#self.pipe.unet.config.cross_attention_dim,
                    ff_mult=4,
                ).to(dtype=torch_dtype)
                # 加载投影层参数
                state_dict = torch.load("/data1/humw/Pretrains/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin", map_location="cpu")
                image_proj_model.load_state_dict(state_dict["image_proj"])
                image_proj_model = image_proj_model.to(args.device, dtype=torch_dtype).eval().requires_grad_(False)
            else:
                image_proj_model = None
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
    if args.target == "non-target":
        target_data = None
    else:
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
        
        trans_512 = [
            transforms.Resize(512, interpolation=resample_interpolation),
            transforms.CenterCrop(512) if args.center_crop else transforms.RandomCrop(512),
        ]
        trans_512 = transforms.Compose(trans_512)
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
            torch_dtype,
            model_dict=model_dict,
            perturbed_images=perturbed_data,
            original_images=original_data,
            target_images=target_data,
            num_steps=args.max_adv_train_steps,
            trans_224=trans_224,
            trans_336=trans_336,
            trans_512=trans_512,
            image_proj_model=image_proj_model,
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
    