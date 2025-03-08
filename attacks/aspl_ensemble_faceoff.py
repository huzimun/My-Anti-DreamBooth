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
from transformers import AutoTokenizer, PretrainedConfig
from utils import LatentAttack, agm
from transformers.models.clip.modeling_clip import CLIPVisionModelWithProjection
import numpy as np
import pdb; 
from ip_adapter.resampler import Resampler

logger = get_logger(__name__)

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


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--untargeted_unet",
        type=int,
        default=0,
        required=True,
        help="1 use untargeted unet, 0 not use untargeted unet",
    )
    parser.add_argument(
        "--fix_unet",
        type=int,
        default=0,
        required=True,
        help="1 use fixed unet, 0 not use",
    )
    parser.add_argument(
        "--fix_noise",
        type=int,
        default=0,
        required=True,
        help="1 use fixed noise, 0 not use",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        default="unet-vae-ipadapter",
        required=True,
        help="unet-vae-ipadapter",
    )
    parser.add_argument(
        "--sds",
        type=int,
        default=0,
        required=True,
        help="1 use sds, 0 not use sds",
    )
    parser.add_argument(
        "--eot",
        type=int,
        default=0,
        required=True,
        help="1 use eot, 0 not use eot",
    )
    parser.add_argument(
        "--agm",
        type=int,
        default=0,
        required=True,
        help="ace or ace-plus",
    )
    parser.add_argument(
        "--w1",
        type=float,
        default=1.0,
        required=False,
        help="weight for unet loss",
    )
    parser.add_argument(
        "--w2",
        type=float,
        default=1.0,
        required=False,
        help="weight for vae loss",
    )
    parser.add_argument(
        "--w3",
        type=float,
        default=1.0,
        required=False,
        help="weight for ipadapter loss",
    )
    parser.add_argument(
        "--w4",
        type=float,
        default=1.0,
        required=False,
        help="weight for ipadaptersd loss",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="ace",
        required=True,
        help="ace or ace-plus",
    )
    parser.add_argument(
        "--distance_choice",
        type=str,
        default="mse",
        required=True,
        help="mse or cosine similarity",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
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
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir_for_train",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_data_dir_for_adversarial",
        type=str,
        default=None,
        required=True,
        help="A folder containing the images to add adversarial noise",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=20,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--max_f_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train surogate model.",
    )
    parser.add_argument(
        "--max_adv_train_steps",
        type=int,
        default=10,
        help="Total number of sub-steps to train adversarial noise.",
    )
    parser.add_argument(
        "--checkpointing_iterations",
        type=int,
        default=5,
        help=("Save a checkpoint of the training state every X iterations."),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
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
    parser.add_argument(
        "--target_image_path",
        default=None,
        help="target image for attacking",
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

def train_one_epoch(
    args,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    num_steps=20,
):
    # Load the tokenizer

    unet, text_encoder = copy.deepcopy(models[0]), copy.deepcopy(models[1])
    params_to_optimize = itertools.chain(unet.parameters(), text_encoder.parameters())

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    train_dataset = DreamBoothDatasetFromTensor(
        data_tensor,
        args.instance_prompt,
        tokenizer,
        args.class_data_dir,
        args.class_prompt,
        args.resolution,
        args.center_crop,
    )

    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    for step in range(num_steps):
        unet.train()
        text_encoder.train()

        step_data = train_dataset[step % len(train_dataset)]
        pixel_values = torch.stack([step_data["instance_images"], step_data["class_images"]]).to(
            device, dtype=weight_dtype
        )
        input_ids = torch.cat([step_data["instance_prompt_ids"], step_data["class_prompt_ids"]], dim=0).to(device)

        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Predict the noise residual
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # with prior preservation loss
        if args.with_prior_preservation:
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

            # Add the prior loss to the instance loss.
            loss = instance_loss + args.prior_loss_weight * prior_loss

        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad()
        print(
            f"Step #{step}, loss: {loss.detach().item()}, prior_loss: {prior_loss.detach().item()}, instance_loss: {instance_loss.detach().item()}"
        )

    return [unet, text_encoder]


def pgd_attack(
    args,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    original_images: torch.Tensor,
    target_tensor: torch.Tensor,
    num_steps: int,
    ipadapter_target_tensor: torch.Tensor,
    ipadapter,
    image_proj_model,
    trans_224,
    trans_512
):
    """Return new perturbed data"""

    unet, text_encoder = models
    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    perturbed_images = data_tensor.detach().clone()
    original_images.requires_grad_(False)
    # perturbed_images.requires_grad_(True)
    if args.target_image_path == "none": # 无目标
        # 加入随机扰动
        perturbed_images = (perturbed_images + (torch.rand(*perturbed_images.shape)*2*args.pgd_eps-args.pgd_eps)).to(weight_dtype)
        
    if "vae" in args.model_types or "unet" in args.model_types or "ipadaptersd" in args.model_types:
        vae.to(device, dtype=weight_dtype)
        if args.target_image_path == "none": # 无目标
            if args.eot == 1:
                original_latents = vae.encode(trans_512(original_images).to(device, dtype=weight_dtype)).latent_dist.sample()
            else:
                original_latents = vae.encode(original_images.to(device, dtype=weight_dtype)).latent_dist.sample()
            original_latents = original_latents * vae.config.scaling_factor # torch.Size([4, 4, 64, 64])
    if "unet" in args.model_types or "ipadaptersd" in args.model_types:
        unet.to(device, dtype=weight_dtype)
        text_encoder.to(device, dtype=weight_dtype)
        
        input_ids = tokenizer( # torch.Size([4, 77])
            args.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.repeat(len(data_tensor), 1)
        
        # sample a random timestep list
        timesteps_list = list()
        for step in range(num_steps):
            # Sample a random timestep for each image
            bsz = perturbed_images.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()
            timesteps_list.append(timesteps)
    if "ipadapter" in args.model_types:
        ipadapter.to(device, dtype=weight_dtype)
        if args.target_image_path == "none": # 无目标 torch.Size([4, 257, 1280])
            ipadapter_original_latents = ipadapter(trans_224(original_images).to(device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                
    pgd_loss_list = list() # 保存损失函数字典
    noise = None # 高斯噪声
    for step in range(num_steps): # 6
        perturbed_images.requires_grad = True
        if "vae" in args.model_types or "unet" in args.model_types or "ipadaptersd" in args.model_types:
            if args.eot == 1:
                latents = vae.encode(trans_512(perturbed_images).to(device, dtype=weight_dtype)).latent_dist.sample()
            else:
                latents = vae.encode(perturbed_images.to(device, dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor # torch.Size([4, 4, 64, 64])
            if args.sds == 1:
                z_latents = latents.clone().detach() # latents不再参与UNet的梯度计算
            else:
                z_latents = latents
        if "unet" in args.model_types or "ipadaptersd" in args.model_types:
            # prepare ipadapter condition
            if image_proj_model is not None:
                num_samples = 1
                # pdb.set_trace()
                ipadapter_perturbed_tensor = ipadapter(trans_224(perturbed_images).to(device), output_hidden_states=True).hidden_states[-2] # torch.Size([4, 257, 1280]
                perturbed_image_prompt_embeds = image_proj_model(ipadapter_perturbed_tensor.to(dtype=torch.float16)).to(dtype=weight_dtype) # torch.Size([4, 16, 768])
                bs_embed, seq_len, _ = perturbed_image_prompt_embeds.shape # 如果传入4张图像，torch.Size([4, 16, 768])
                perturbed_image_prompt_embeds = perturbed_image_prompt_embeds.repeat(1, num_samples, 1) # 如果传入4张图像，torch.Size([4, 64, 768]),num_samples=1时torch.Size([4, 16, 768])
                perturbed_image_prompt_embeds = perturbed_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1) # torch.Size([4, 16, 768])，如果传入4张图像，torch.Size([16, 16, 768]),num_samples=1时torch.Size([4, 16, 768])
            else:
                ipadapter_perturbed_tensor = None
            # Sample noise that we'll add to the latents
            if not (noise is not None and args.fix_noise == 1): # 使用固定的高斯噪声
                noise = torch.randn_like(latents)
            bsz = latents.shape[0] # 4
            # Sample a random timestep for each image
            # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            # timesteps = timesteps.long()
            timesteps = timesteps_list[step]
            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(z_latents, noise, timesteps) # torch.Size([4, 4, 64, 64])

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(input_ids.to(device))[0] # torch.Size([4, 77, 768])

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample # torch.Size([4, 4, 64, 64])
            
            # predict the noise residual for with ipadapter condiition
            # ipadapter_latents = torch.randn_like(latents)
            # 这一步应该和IP-Adapter实际的预测过程不一致，至少latents就不是同样的计算方式
            if image_proj_model is not None:
                ipadapter_model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=perturbed_image_prompt_embeds).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(z_latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            unet.zero_grad()
            text_encoder.zero_grad()
            if args.sds == 1:
                # update z
                grad = (model_pred - noise)
                latents.backward(gradient=grad) # 手动指定了在反向传播过程中，相对于latents的梯度
                g_x = perturbed_images.grad.detach() # 提前计算好梯度
                perturbed_images = perturbed_images.clone().detach() # 非常重要，否则无法再次计算vae的梯度
                perturbed_images.requires_grad_(True)
                if args.eot == 1:
                    latents = vae.encode(trans_512(perturbed_images).to(device, dtype=weight_dtype)).latent_dist.sample()
                else:
                    latents = vae.encode(perturbed_images.to(device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor # torch.Size([4, 4, 64, 64])

            if args.distance_choice == "mse":
                loss_untargeted_unet = F.mse_loss(model_pred.to(weight_dtype), target.to(weight_dtype), reduction="mean")
                if image_proj_model is not None:
                    loss_untargeted_ipadaptersd = F.mse_loss(ipadapter_model_pred.to(weight_dtype), target.to(weight_dtype), reduction="mean")
            else: # cosine
                loss_untargeted_unet = - F.cosine_similarity(model_pred.to(weight_dtype), target.to(weight_dtype), -1).mean()
                if image_proj_model is not None:
                    loss_untargeted_ipadaptersd = - F.cosine_similarity(ipadapter_model_pred.to(weight_dtype), target.to(weight_dtype), -1).mean()
        else: # "unet" not in args.model_types
            encoder_hidden_states = None
            timesteps = None
        # target-shift loss
        if target_tensor is not None or ipadapter_target_tensor is not None or args.mode == "ace-plus-ipadapter":
            if args.mode == 'ace':
                loss_unet = - F.mse_loss(model_pred.to(weight_dtype), target_tensor.to(weight_dtype))
                loss = loss_unet
                print(f"PGD loss - step {step}, loss: {loss.detach().item()}, loss_unet: {loss_unet.detach().item()}")
                loss_dict = {'loss': loss.detach().item(), 'loss_unet': loss_unet.detach().item()}
                pgd_loss_list.append(loss_dict)
            elif args.mode == 'ace-plus':
                # latent_attack = LatentAttack()
                # loss = loss - 1e2 * latent_attack(latents.float(), target_tensor=target_tensor.float())   
                loss_unet = - F.mse_loss(model_pred.to(weight_dtype), target_tensor.to(weight_dtype))
                loss_vae = - F.mse_loss(latents.to(weight_dtype), target_tensor.to(weight_dtype), reduction="mean")
                w1 = 1
                w2 = 100
                loss = w1* loss_unet + w2 * loss_vae  
                print(f"PGD loss - step {step}, loss: {loss.detach().item()}, loss_unet: {loss_unet.detach().item()}, loss_vae: {loss_vae.detach().item()}")
                loss_dict = {'loss': loss.detach().item(), 'loss_unet': loss_unet.detach().item(), 'loss_vae': loss_vae.detach().item()}
                pgd_loss_list.append(loss_dict)
            elif args.mode == "ace-plus-ipadapter":
                loss_unet = 0
                loss_vae = 0
                loss_ipadapter = 0
                loss_ipadaptersd = 0
                # pdb.set_trace()
                model_types = args.model_types.split('-')
                if args.distance_choice == "mse":
                    if "unet" in model_types:
                        if args.untargeted_unet == 1: # 使用无目标unet损失
                            unet_target_tensor = target
                            loss_unet = loss_untargeted_unet
                        elif target_tensor is not None:
                            unet_target_tensor = target_tensor
                            loss_unet = - F.mse_loss(model_pred.to(weight_dtype), target_tensor.to(weight_dtype))
                        else: # 其实就是无目标
                            unet_target_tensor = target
                            loss_unet = F.mse_loss(model_pred.to(weight_dtype), target.to(weight_dtype)) # max
                    if "vae" in model_types:
                        if target_tensor is not None:
                            vae_target_tensor = target_tensor
                            loss_vae = - F.mse_loss(latents.to(weight_dtype), target_tensor.to(weight_dtype), reduction="mean")
                        else:
                            vae_target_tensor = original_latents
                            loss_vae = F.mse_loss(latents.to(weight_dtype), original_latents.to(weight_dtype), reduction="mean")
                    if "ipadapter" in model_types:
                        # if ipadapter_perturbed_tensor is None: # 如果前面ipadaptersd分支计算过了
                        ipadapter_perturbed_tensor = ipadapter(trans_224(perturbed_images).to(device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2] # torch.Size([4, 257, 1280])
                        ipadapter.zero_grad()
                        if ipadapter_target_tensor is not None:
                            # ipadapter_target_tensor = ipadapter_target_tensor
                            loss_ipadapter = - F.mse_loss(ipadapter_perturbed_tensor.to(weight_dtype), ipadapter_target_tensor.to(weight_dtype)) # -min=max
                        else:
                            ipadapter_target_tensor = ipadapter_original_latents
                            loss_ipadapter = F.mse_loss(ipadapter_perturbed_tensor.to(weight_dtype), ipadapter_original_latents.to(weight_dtype))
                    if "ipadaptersd" in model_types:
                        if args.untargeted_unet == 1: # 使用无目标unet损失
                            ipadaptersd_target_tensor = target
                            loss_ipadaptersd = loss_untargeted_ipadaptersd
                        elif target_tensor is not None:
                            ipadaptersd_target_tensor = target_tensor
                            loss_ipadaptersd = - F.mse_loss(ipadapter_model_pred.to(weight_dtype), target_tensor.to(weight_dtype))
                        else: # 其实就是无目标
                            ipadaptersd_target_tensor = target
                            loss_ipadaptersd = F.mse_loss(ipadapter_model_pred.to(weight_dtype), target.to(weight_dtype))
                            
                else: # cosine
                    if "unet" in model_types:
                        if args.untargeted_unet == 1: # 使用无目标unet损失
                            # unet_target_tensor = target
                            loss_unet = loss_untargeted_unet
                        elif target_tensor is not None:
                            unet_target_tensor = target_tensor
                            loss_unet = F.cosine_similarity(model_pred.to(weight_dtype), target_tensor.to(weight_dtype), -1).mean()
                        else:
                            unet_target_tensor = target
                            loss_unet = - F.cosine_similarity(model_pred.to(weight_dtype), target.to(weight_dtype), -1).mean()
                    if "vae" in model_types:
                        if target_tensor is not None:
                            vae_target_tensor = target_tensor
                            loss_vae = F.cosine_similarity(latents.to(weight_dtype), target_tensor.to(weight_dtype), -1).mean()
                        else:
                            vae_target_tensor = original_latents
                            loss_vae = - F.cosine_similarity(latents.to(weight_dtype), original_latents.to(weight_dtype), -1).mean()
                    if "ipadapter" in model_types:
                        # if ipadapter_perturbed_tensor is None: # 前面ipadaptersd分支已经计算过了
                        ipadapter_perturbed_tensor = ipadapter(trans_224(perturbed_images).to(device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                        ipadapter.zero_grad()
                        if ipadapter_target_tensor is not None:
                            loss_ipadapter = F.cosine_similarity(ipadapter_perturbed_tensor.to(weight_dtype), ipadapter_target_tensor.to(weight_dtype), -1).mean()
                        else:
                            loss_ipadapter = - F.cosine_similarity(ipadapter_perturbed_tensor.to(weight_dtype), ipadapter_original_latents.to(weight_dtype), -1).mean()
                    if "ipadaptersd" in model_types:
                        if args.untargeted_unet == 1: # 使用无目标unet损失
                            ipadaptersd_target_tensor = target
                            loss_ipadaptersd = loss_untargeted_ipadaptersd
                        elif target_tensor is not None:
                            ipadaptersd_target_tensor = target_tensor
                            loss_ipadaptersd = F.cosine_similarity(ipadapter_model_pred.to(weight_dtype), target_tensor.to(weight_dtype), -1).mean()
                        else:
                            ipadaptersd_target_tensor = target
                            loss_ipadaptersd = - F.cosine_similarity(ipadapter_model_pred.to(weight_dtype), target.to(weight_dtype), -1).mean()
                w1 = args.w1
                w2 = args.w2
                w3 = args.w3
                w4 = args.w4
                
                if args.agm == 1:
                    # models = [unet, vae, ipadapter]
                    # losses = [loss_unet, loss_vae, loss_ipadapter]
                    # grads = [torch.autograd.grad(losses[idx], perturbed_images, retain_graph=True, create_graph=False)[0] for idx in range(len(models))]
                    # targets = [target_tensor.to(weight_dtype), target_tensor.to(weight_dtype), ipadapter_target_tensor.to(weight_dtype)]
                    models_dict = {}
                    losses_dict = {}
                    grads_dict = {}
                    targets_dict = {}
                    for model_type in model_types:
                        if model_type == 'unet':
                            models_dict[model_type] = unet
                            losses_dict[model_type] = loss_unet
                            if target_tensor is not None:
                                targets_dict[model_type] = unet_target_tensor.to(weight_dtype)
                        elif model_type == 'vae':
                            models_dict[model_type] = vae
                            losses_dict[model_type] = loss_vae
                            if target_tensor is not None:
                                targets_dict[model_type] = vae_target_tensor.to(weight_dtype)
                        elif model_type == 'ipadapter':
                            models_dict[model_type] = ipadapter
                            losses_dict[model_type] = loss_ipadapter
                            if ipadapter_target_tensor is not None:
                                targets_dict[model_type] = ipadapter_target_tensor.to(weight_dtype)
                        elif model_type == 'ipadaptersd':
                            models_dict[model_type] = image_proj_model # 有ipadaptersd，必然有clip和unet，这里只需要加载映射层
                            losses_dict[model_type] = loss_ipadaptersd
                            if ipadaptersd_target_tensor is not None:
                                targets_dict[model_type] = ipadaptersd_target_tensor.to(weight_dtype)
                        else:
                            raise NotImplementedError
                        if len(model_types) > 1:
                            grad = torch.autograd.grad(losses_dict[model_type], perturbed_images, retain_graph=True, create_graph=False)[0]
                            grads_dict[model_type] = grad
                    if "unet" in model_types and "vae" not in model_types: # 即便不攻击vae，unet也需要用到vae
                        models_dict["vae"] = vae
                    if "ipadaptersd" in model_types:
                        if "ipadapter" not in model_types:
                            models_dict["ipadapter"] = ipadapter
                        if "unet" not in model_types:
                            models_dict["unet"] = unet
                    # AGM
                    if len(model_types) > 1:
                        alphas = agm(args, ori_data=original_images, cur_adv=perturbed_images, grads_dict=grads_dict, targets_dict=targets_dict, beta=10,
                            models_dict=models_dict, timesteps=timesteps, trans_512=trans_512, trans_224=trans_224,
                            noise_scheduler=noise_scheduler, device=device, weight_dtype=weight_dtype, 
                            encoder_hidden_states=encoder_hidden_states, model_types=model_types)
                    else:
                        alphas = torch.ones(len(model_types)).to(device)
                    print("alpha: ", alphas)
                    # grad_list = [grads_dict["unet"].float(), grads_dict["vae"].float(), grads_dict["ipadapter"].float()]
                    # # 定义一个3*3 numpy矩阵，矩阵的每个元素是grd_list[i]和grad_list[j]的余弦相似度分数
                    # cos_sim_matrix = np.zeros((3, 3))
                    # for i in range(3):
                    #     for j in range(3):
                    #         cos_sim_matrix[i][j] = F.cosine_similarity(grad_list[i], grad_list[j], -1).mean()
                    # print("cos_sim_matrix: \n", cos_sim_matrix)
                    # alphas_grad_list = [(grads_dict["unet"] * alphas[0].cpu()).float(), (grads_dict["vae"] * alphas[1].cpu()).float(), (grads_dict["ipadapter"] * alphas[2].cpu()).float()]
                    # alphas_cos_sim_matrix = np.zeros((3, 3))
                    # for i in range(3):
                    #     for j in range(3):
                    #         alphas_cos_sim_matrix[i][j] = F.cosine_similarity(alphas_grad_list[i], alphas_grad_list[j], -1).mean()
                    # print("alphas_cos_sim_matrix: \n", alphas_cos_sim_matrix)
                    # loss = alphas[0] * loss_unet + alphas[1] * loss_vae + alphas[2] * loss_ipadapter
                    loss = 0.0
                    loss_sum = 0.0
                    for j, model_type in enumerate(model_types):
                        loss += alphas[j] * losses_dict[model_type]
                        loss_sum += losses_dict[model_type]
                else:
                    if args.sds == 1:
                        # encoder_loss = w2 * loss_vae + w3 * loss_ipadapter
                        # vae.zero_grad()
                        # ipadapter.zero_grad()
                        # encoder_loss.backward()
                        # grad = torch.autograd.grad(encoder_loss, perturbed_images)[0]
                        g_vae = torch.autograd.grad(loss_vae, perturbed_images)[0]
                        g_ipadapter = torch.autograd.grad(loss_ipadapter, perturbed_images)[0]
                        # g_tex = grad.detach()
                    loss = w1 * loss_unet + w2 * loss_vae + w3 * loss_ipadapter + w4 * loss_ipadaptersd
                    loss_sum = loss_unet + loss_vae + loss_ipadapter + loss_ipadaptersd
                    losses_dict = {"unet": loss_unet, "vae": loss_vae, "ipadapter": loss_ipadapter, "ipadaptersd": loss_ipadaptersd}

                print(f"PGD loss - step {step}, loss_sum: {loss_sum.detach().item()}, loss: {loss.detach().item()}")
                loss_dict = {"loss_sum": loss_sum.detach().item(), 'loss': loss.detach().item()}
                for model_type in model_types:
                    loss_dict[model_type] = losses_dict[model_type].detach().item()
                    print(f"{model_type}: {losses_dict[model_type].detach().item()}")
                pgd_loss_list.append(loss_dict)
            else:         
                xtm1_pred = torch.cat(
                    [
                        noise_scheduler.step(
                            model_pred[idx : idx + 1],
                            timesteps[idx : idx + 1],
                            noisy_latents[idx : idx + 1],
                        ).prev_sample
                        for idx in range(len(model_pred))
                    ]
                )
                xtm1_target = noise_scheduler.add_noise(target_tensor, noise, timesteps - 1)
                loss = loss - F.mse_loss(xtm1_pred, xtm1_target)
        else: # not (target_tensor is not None or ipadapter_target_tensor is not None or args.mode == "ace-plus-ipadapter")
            loss = loss_untargeted_unet
        if args.sds == 0: # 如果使用了sds，就不必计算这一步，因为已经计算过梯度了
            grad = torch.autograd.grad(loss, perturbed_images)[0]
        else:
            grad = w1 * g_x + w2 * g_vae + w3 * g_ipadapter # 编码器损失已经预先加权过了
        # loss.backward()
        
        # alpha = args.pgd_alpha
        # eps = args.pgd_eps

        # adv_images = perturbed_images + args.pgd_alpha * perturbed_images.grad.sign()
        adv_images = perturbed_images + grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-args.pgd_eps, max=+args.pgd_eps)
        if args.eot == 0:
            perturbed_images = torch.clamp(original_images + eta, min=torch.min(original_images), max=torch.max(original_images)).detach_()
        else:
            perturbed_images = torch.clamp(original_images + eta, min=torch.min(original_images), max=torch.max(original_images)).detach().clone()
        # print(f"PGD loss - step {step}, loss: {loss.detach().item()}")
    return perturbed_images, pgd_loss_list


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
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
    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if args.mixed_precision == "fp32":
                torch_dtype = torch.float32
            elif args.mixed_precision == "fp16":
                torch_dtype = torch.float16
            elif args.mixed_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader,
                desc="Generating class images",
                disable=not accelerator.is_local_main_process,
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    if "unet" in args.model_types or "ipadaptersd" in args.model_types:
        
        # Load scheduler and models
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
        )
        if not args.train_text_encoder:
            text_encoder.requires_grad_(False)
        
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    else:
        text_encoder = None
        unet = None
        tokenizer = None
        noise_scheduler = None

    if "unet" in args.model_types or "vae" in args.model_types or "ipadaptersd" in args.model_types:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        ).cuda().to(dtype=torch_dtype)

        vae.requires_grad_(False)
    else:
        vae = None

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    if (args.w1 != 0 or args.w2 != 0) and "unet" in args.model_types:
        clean_data = load_data(
            args,
            data_dir=args.instance_data_dir_for_train,
            size=args.resolution,
            center_crop=args.center_crop,
        )
    perturbed_data = load_data(
        args,
        data_dir=args.instance_data_dir_for_adversarial,
        size=args.resolution,
        center_crop=args.center_crop,
    )
    original_data = perturbed_data.clone()
    original_data.requires_grad_(False)

    
    if "ipadapter" in args.model_types: # 不管是只有ipadapter还是有ipadaptersd，都需要加载CLIP图像编码器
        ipadapter_path = "/data1/humw/Pretrains/IP-Adapter/models/image_encoder"
        ipadapter = CLIPVisionModelWithProjection.from_pretrained(ipadapter_path).cuda().to(dtype=torch_dtype)
        if "ipadaptersd" in args.model_types:
            num_tokens = 16
            image_proj_model = Resampler(
                dim=unet.config.cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=num_tokens,
                embedding_dim=ipadapter.config.hidden_size,
                output_dim=unet.config.cross_attention_dim,
                ff_mult=4,
            ).to("cuda", dtype=torch_dtype)
            ip_ckpt_path = "/data1/humw/Pretrains/IP-Adapter/models/ip-adapter-plus_sd15.bin"
            state_dict = torch.load(ip_ckpt_path, map_location="cpu")
            image_proj_model.load_state_dict(state_dict["image_proj"])
        else:
            image_proj_model = None
    else:
        ipadapter = None
    
    resample_interpolation = transforms.InterpolationMode.BILINEAR
    center_crop = True
    if args.eot == 0: # 不用eot，不过ipadapter因为尺寸不一致不得不用eot
        trans_224 = [
            transforms.Resize(224, interpolation=resample_interpolation),
            transforms.CenterCrop(224) if center_crop else transforms.RandomCrop(224),
        ]
        trans_224 = transforms.Compose(trans_224)
        trans_512 = None
    else: # 使用eot
        resample_interpolation = transforms.InterpolationMode.BILINEAR
            
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
        
        train_aug_512 = [
            transforms.Resize(512, interpolation=resample_interpolation),
            transforms.CenterCrop(512) if args.center_crop else transforms.RandomCrop(512),
        ]
        trans_512 = train_aug_512 + tensorize_and_normalize
        trans_512 = transforms.Compose(trans_512)
        print("all_trans:{}".format(trans_512))
        args.pgd_eps = 16.0
        args.pgd_alpha = 16/10 # 默认是阈值的1/10
    target_latent_tensor = None
    if args.target_image_path is not None and args.target_image_path != "none":
        if "vae" in args.model_types or "unet" in args.model_types or "ipadaptersd" in args.model_types:
            target_image_path = Path(args.target_image_path)
            assert target_image_path.is_file(), f"Target image path {target_image_path} does not exist"

            target_image = Image.open(target_image_path).convert("RGB").resize((args.resolution, args.resolution))
            target_image = np.array(target_image)[None].transpose(0, 3, 1, 2)
            if args.eot == 0:
                target_image_tensor = torch.from_numpy(target_image).to("cuda", dtype=torch_dtype) / 127.5 - 1.0
                target_latent_tensor = (
                    vae.encode(target_image_tensor).latent_dist.sample().to(dtype=torch_dtype) * vae.config.scaling_factor
                )
            else:
                target_image_tensor = torch.from_numpy(target_image).to("cuda", dtype=torch_dtype)
                target_latent_tensor = (
                    vae.encode(trans_512(target_image_tensor)).latent_dist.sample().to(dtype=torch_dtype) * vae.config.scaling_factor
                )
            
            target_latent_tensor = target_latent_tensor.repeat(len(perturbed_data), 1, 1, 1).cuda()
        else:
            target_latent_tensor = None
        if "ipadapter" in args.model_types:
            if target_latent_tensor is None:
                target_image_path = Path(args.target_image_path)
                assert target_image_path.is_file(), f"Target image path {target_image_path} does not exist"

                target_image = Image.open(target_image_path).convert("RGB").resize((args.resolution, args.resolution))
                target_image = np.array(target_image)[None].transpose(0, 3, 1, 2)
                if args.eot == 0:
                    target_image_tensor = torch.from_numpy(target_image).to("cuda", dtype=torch_dtype) / 127.5 - 1.0
                else:
                    target_image_tensor = torch.from_numpy(target_image).to("cuda", dtype=torch_dtype)
            # get ipadapter target latent tensor
            ipadapter_target_tensor = ipadapter(trans_224(target_image_tensor), output_hidden_states=True).hidden_states[-2].detach() # torch.Size([1, 257, 1280])
            ipadapter_target_tensor = ipadapter_target_tensor.repeat(len(perturbed_data), 1, 1).cuda() # torch.Size([4, 257, 1280])
        else:
            ipadapter_target_tensor = None
        target_image_tensor = target_image_tensor.to('cpu')
        del target_image_tensor
    else:
        target_latent_tensor = None
        ipadapter_target_tensor = None

    pgd_loss_list = []
    f = [unet, text_encoder]
    for i in range(args.max_train_steps):
        # 1. f' = f.clone()
        f_sur = copy.deepcopy(f)
        if (args.w1 != 0 or args.w2 != 0) and "unet" in args.model_types and args.fix_unet == 0: # unet和vae loss都不等于0，且代理模型包括unet，就训练dreambooth
            f_sur = train_one_epoch(
                args,
                f_sur,
                tokenizer,
                noise_scheduler,
                vae,
                clean_data,
                args.max_f_train_steps,
            )
        perturbed_data, tmp_pgd_loss_list  = pgd_attack(
            args,
            models=f_sur,
            tokenizer=tokenizer,
            noise_scheduler=noise_scheduler,
            vae=vae,
            data_tensor=perturbed_data,
            original_images=original_data,
            target_tensor=target_latent_tensor,
            num_steps=args.max_adv_train_steps,
            ipadapter_target_tensor=ipadapter_target_tensor,
            ipadapter=ipadapter,
            image_proj_model=image_proj_model,
            trans_224=trans_224,
            trans_512=trans_512,
        )
        if (args.w1 != 0 or args.w2 != 0) and "unet" in args.model_types and args.fix_unet == 0: # unet和vae loss都等于0，就不再训练dreambooth了
            f = train_one_epoch(
                args,
                f,
                tokenizer,
                noise_scheduler,
                vae,
                perturbed_data,
                args.max_f_train_steps,
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
        f.write(person_id + '\n')
        f.write(str(pgd_loss_list) + "\n")
        for index, loss_dict in enumerate(pgd_loss_list):
            f.write("index: " + str(index) + ", " + str(loss_dict) + "\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)
    