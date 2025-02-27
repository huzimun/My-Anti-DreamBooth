import torch
from PIL import Image
import os
from ip_adapter import IPAdapterPlusXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
from pathlib import Path
import argparse
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from ip_adapter import IPAdapterPlus
import os
from typing import List, Tuple
import inspect
import torch
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from typing import Any, Callable, Dict, List, Optional, Union
from transformers import AutoTokenizer, PretrainedConfig
from tqdm import tqdm
from ip_adapter.utils import is_torch2_available, get_generator
import torch.nn.functional as F
import pdb
from torchvision import transforms


if is_torch2_available():
    from ip_adapter.attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from ip_adapter.attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from ip_adapter.attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from ip_adapter.resampler import Resampler

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

def prepare_extra_step_kwargs(generator, eta, step):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
def check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    if (callback_steps is None) or (
        callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
    ):
        raise ValueError(
            f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
            f" {type(callback_steps)}."
        )

    if prompt is not None and prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
            " only forward one of the two."
        )
    elif prompt is None and prompt_embeds is None:
        raise ValueError(
            "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
        )
    elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
        raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError(
            f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
            f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
        )

    if prompt_embeds is not None and negative_prompt_embeds is not None:
        if prompt_embeds.shape != negative_prompt_embeds.shape:
            raise ValueError(
                "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                f" {negative_prompt_embeds.shape}."
            )

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
                
def encode_prompt(
        args,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
            prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
            Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
    """
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=torch.float16,
        use_fast=False,
    )
    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, revision=torch.float16)

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=torch.float16,
    ).to(args.device, dtype=torch.float16)
    if prompt_embeds is None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        max_length = prompt_embeds.shape[1]
        uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        negative_prompt_embeds = text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        # prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    del text_encoder, tokenizer
    return prompt_embeds, negative_prompt_embeds
    
def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    # if generator is not None:
    #     gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
    #     if gen_device_type != device.type and gen_device_type == "cpu":
    #         rand_device = "cpu"
    #         # if device != "mps":
    #         #     logger.info(
    #         #         f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
    #         #         f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
    #         #         f" slighly speed up this function by passing a generator that was created on the {device} device."
    #         #     )
    #     elif gen_device_type != device.type and gen_device_type == "cuda":
    #         raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents, vae_scale_factor, init_noise_sigma):
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * init_noise_sigma
    return latents

def load_data(data_dir, image_size=224, resample=2):
    import numpy as np
    def image_to_numpy(image):
        return np.array(image).astype(np.uint8)
    # more robust loading to avoid loaing non-image files
    images = [] 
    for i in list(Path(data_dir).iterdir()):
        if not i.suffix in [".jpg", ".png", ".jpeg"]:
            continue
        else:
            images.append(image_to_numpy(Image.open(i).convert("RGB")))
    # resize the images to 512 x 512, resample value 2 means BILINEAR
    images = [Image.fromarray(i).resize((image_size, image_size), resample) for i in images]
    # copy from metacloak
    images = np.stack(images)
    # from B x H x W x C to B x C x H x W
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    # images = np.array(images).transpose(0, 3, 1, 2)
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
    
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example for white-box attack")
    parser.add_argument(
        "--model_type",
        type=str,
        default="sd15",
        required=True,
        help="sd15 or sdxl",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=False,
        help="Path to input folders",
    )
    parser.add_argument(
        "--sub_name",
        type=str,
        default="set_B",
        required=True,
        help=("subfolder name of input dir"),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/humw/Pretrain/RealVisXL_V3.0",
        required=False,
        help=("/home/humw/Pretrain/RealVisXL_V3.0"),
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="/home/humw/Pretrain/h94/IP-Adapter/models/image_encoder",
        required=False,
        help = "hflip or not"
    )
    parser.add_argument(
        "--ip_ckpt",
        type=str,
        default="/home/humw/Pretrain/h94/IP-Adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin",
        required=False,
        help = "ip ckpt"
    )
    parser.add_argument(
        "--vae_model_path",
        type=str,
        default="/data1/humw/Pretrains/sd-vae-ft-mse",
        required=False,
        help = "vae_model_path"
    )
    parser.add_argument(
        "--target_image_path",
        type=str,
        default="/home/humw/Codes/FaceOff/output/Exp1/ipadapter/min-VGGFace2_ipadapter_out-224_no-mid-size_loss-n-mse_alpha6_eps16_num200_pre-test",
        required=False,
        help = "target_image_path dir"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/humw/Codes/FaceOff/output/Exp1/ipadapter/min-VGGFace2_ipadapter_out-224_no-mid-size_loss-n-mse_alpha6_eps16_num200_pre-test",
        required=False,
        help = "input dir"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/humw/Codes/FaceOff/target_model/output/ipadapter/min-VGGFace2_ipadapter_out-224_no-mid-size_loss-n-mse_alpha6_eps16_num200_pre-test",
        required=False,
        help = "output dir"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        required=False,
        help = "image resolution of target model, clip is 224"
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default="bf16",
        required=False,
        help = "bf16 is dtype=torch.float16"
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
    
def main(args):
    print(args)
    # import pdb; pdb.set_trace()
    if args.prior_generation_precision == "fp32":
        torch_dtype = torch.float32
    elif args.prior_generation_precision == "fp16":
        torch_dtype = torch.float16
    elif args.prior_generation_precision == "bf16":
        torch_dtype = torch.float16
    else:
        raise ValueError("prior_generation_precision must be one of [fp32, fp16, bf16]")
    # init
    num_tokens = 16
    num_samples=1 # 4张图像的时候设置为1，否则会爆显存
    seed=42
    guidance_scale=7.5
    num_inference_steps=30
    scale=1.0
    
    vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(dtype=torch_dtype)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(args.device, dtype=torch_dtype)
    # 对应ip_adapter.py中的self.set_scale(scale)
    for attn_processor in unet.attn_processors.values():
        if isinstance(attn_processor, IPAttnProcessor):
            attn_processor.scale = scale
    # pdb.set_trace()
    # set_ip_adapter()
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=1.0,
                num_tokens=num_tokens,
            ).to(args.device, dtype=torch.float16)
    unet.set_attn_processor(attn_procs)
    # pdb.set_trace()
    # load image encoder
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path).to(
        args.device, dtype=torch.float16
    )
    # clip_image_processor = CLIPImageProcessor()
    resample_interpolation = transforms.InterpolationMode.BILINEAR
    center_crop = True
    train_aug = [
        transforms.Resize(size=224, interpolation=resample_interpolation),
        transforms.CenterCrop(size=224) if center_crop else transforms.RandomCrop(size=224),
    ]
    tensorize_and_normalize = [
        transforms.Normalize([0.5*255]*3,[0.5*255]*3),
    ]
    all_trans = train_aug + tensorize_and_normalize
    all_trans = transforms.Compose(all_trans)
    print("all_trans:{}".format(all_trans))
        
    # 读取target_image
    
    target_images = load_data(args.target_image_path, image_size=512, ) # 之所以不用224，是因为要输入vae
    target_latent_tensor = (
        vae.encode(target_images).latent_dist.sample().to(dtype=torch_dtype) * vae.config.scaling_factor
    )
    
    # image proj model, init_proj()
    image_proj_model = Resampler(
            dim=unet.config.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=12,
            num_queries=num_tokens,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=unet.config.cross_attention_dim,
            ff_mult=4,
    ).to(args.device, dtype=torch.float16)
    # load_ip_adapter()
    if os.path.splitext(args.ip_ckpt)[-1] == ".safetensors":
        state_dict = {"image_proj": {}, "ip_adapter": {}}
        with safe_open(args.ip_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("image_proj."):
                    state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                elif key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
    else:
        state_dict = torch.load(args.ip_ckpt, map_location="cpu")
    # pdb.set_trace()
    image_proj_model.load_state_dict(state_dict["image_proj"])
    ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
    ip_layers.load_state_dict(state_dict["ip_adapter"])
    
    # 0. Default height and width to unet
    height = 512 or unet.config.sample_size * vae_scale_factor
    width = 512 or unet.config.sample_size * vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    prompt="a photo of person"
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality"
    

    if prompt is None:
        prompt = "best quality, high quality"
    if negative_prompt is None:
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    
    callback_steps = 1
    check_inputs(
        prompt, height, width, callback_steps, negative_prompt
    )
    
    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0
    # pdb.set_trace()
    for person_id in sorted(os.listdir(args.input_dir)):
        print(person_id)
        person_dir = os.path.join(args.input_dir, person_id + "/" + args.sub_name)
        print(person_dir)
        original_images = load_data(person_dir, args.resolution, 2) # 4张图像，torch.tensor, torch.Size([4, 3, 512, 512])
        original_images = original_images.to(args.device, dtype=torch.float16)
        perturbed_images = original_images.detach().clone() # torch.Size([4, 3, 512, 512])
        perturbed_images = perturbed_images.to(args.device, dtype=torch.float16)
        
        num_prompts = 1 if isinstance(original_images, Image.Image) else len(original_images)

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        # Get image embeds
        # image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds( # torch.Size([1, 16, 768]), torch.Size([1, 16, 768])
        #     pil_image=pil_image, clip_image_embeds=clip_image_embeds
        # )

        # 4. Prepare timesteps
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        scheduler.set_timesteps(num_inference_steps, device=args.device)
        timesteps = scheduler.timesteps
        # pdb.set_trace()
        # 5. Prepare latent variables
        generator = get_generator(seed, args.device)
        num_channels_latents = unet.in_channels
        num_images_per_prompt = 1
        batch_size = 4
        latents = prepare_latents( # latents就是初始noise，逐步减噪使得其接近真实图像,torch.Size([16, 4, 64, 64])，将num_samples设置为1后torch.Size([4, 4, 64, 64])
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch_dtype,
            args.device,
            generator,
            latents=None,
            vae_scale_factor=vae_scale_factor,
            init_noise_sigma=scheduler.init_noise_sigma
        )
        # noises = latents.clone().detach() # 用于计算损失函数，目的是使得条件输出接近随机噪声,torch.Size([16, 4, 64, 64])，将num_samples设置为1后torch.Size([4, 4, 64, 64])

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        eta = 0.0
        extra_step_kwargs = prepare_extra_step_kwargs(generator, eta, scheduler.step)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
        pbar = tqdm(total=num_inference_steps, desc="attack models")
        # pdb.set_trace()
        for i, t in enumerate(timesteps):
            perturbed_images.requires_grad_() # 允许保护图像获取梯度
            tran_perturbed_images = all_trans(perturbed_images) # torch.Size([4, 3, 224, 224])
            # tran_perturbed_images = tran_perturbed_images.to(args.device, dtype=torch.float16) # torch.Size([4, 3, 224, 224])
            clip_image_embeds = image_encoder(tran_perturbed_images, output_hidden_states=True).hidden_states[-2] # torch.Size([4, 257, 1280])
            image_prompt_embeds = image_proj_model(clip_image_embeds) # torch.Size([4, 16, 768])
            uncond_clip_image_embeds = image_encoder( # torch.Size([4, 257, 1280])
                torch.zeros_like(tran_perturbed_images), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_prompt_embeds = image_proj_model(uncond_clip_image_embeds) # torch.Size([4, 16, 768])
            # pdb.set_trace()
            bs_embed, seq_len, _ = image_prompt_embeds.shape # 如果传入4张图像，torch.Size([4, 16, 768])
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1) # 如果传入4张图像，torch.Size([4, 64, 768]),num_samples=1时torch.Size([4, 16, 768])
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1) # torch.Size([4, 16, 768])，如果传入4张图像，torch.Size([16, 16, 768]),num_samples=1时torch.Size([4, 16, 768])
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1) # torch.Size([1, 64, 768])，4张图像，torch.Size([4, 64, 768])
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1) # torch.Size([4, 16, 768])，4张图像，torch.Size([16, 16, 768])
            # pdb.set_trace()
            # 3. Encode input prompt
            with torch.inference_mode():
                prompt_embeds_, negative_prompt_embeds_ = encode_prompt( # torch.Size([4, 77, 768])
                    args,
                    prompt,
                    device=args.device,
                    num_images_per_prompt=num_samples,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )
            # prompt_embeds_.requires_grad_() # 叶子节点
            # TODO：最大的麻烦是，torch.cat之后得到的prompt_embeds也是叶子节点，梯度传递在这里就断了
            # prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1) # torch.Size([4, 93, 768]),16+77=93
            prompt_embeds = image_prompt_embeds
            noises = torch.randn_like(latents)
            latents = scheduler.scale_model_input(latents, t) # torch.Size([32, 4, 64, 64])
            # pdb.set_trace()
            # predict the noise residual
            noise_pred = unet( # torch.Size([4, 4, 64, 64])
                latents, # torch.Size([32, 4, 64, 64]), 没有negative prompt的时候是torch.Size([16, 4, 64, 64])，num_samples=1的时候是torch.Size([4, 4, 64, 64])
                t,
                encoder_hidden_states=prompt_embeds, # torch.Size([32, 93, 768]), 没有negative prompt的时候是torch.Size([16, 93, 768]),num_samples=1的时候是torch.Size([4, 93, 768])
            ).sample
            
            # 预测的噪声和真实噪声之间的损失，最大化，加上梯度
            loss = F.mse_loss(noise_pred, noises, reduction="mean")
            print(f"loss: {loss.detach().item()}")
            # loss.backward()
            alpha = args.pgd_alpha
            eps = args.pgd_eps
            # pdb.set_trace()
            grad = torch.autograd.grad(loss, perturbed_images)[0]
            adv_images = perturbed_images + alpha * grad.sign()
            eta = torch.clamp(adv_images - original_images, min=-eps, max=+eps)
            perturbed_images = torch.clamp(original_images + eta, min=torch.min(original_images), max=torch.max(original_images)).detach_()
            print(f"PGD loss - step {t}, loss: {loss.detach().item()}")
            # compute the previous noisy sample x_t -> x_t-1
            # TODO: 这个latents是否需要更新，还需要确认
            latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample # torch.Size([4, 4, 64, 64])
            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                pbar.update(1)
                callback = None
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
        # 确认args.output_dir是否存在，如果不存在则创建
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        person_id_output_dir = os.path.join(args.output_dir, person_id)
        if not os.path.exists(person_id_output_dir):
            os.makedirs(person_id_output_dir)
        # 保存图片到person_id_output_dir目录下
        save_image(person_id_output_dir, person_dir, perturbed_images)
        
    #     pdb.set_trace()
    # pdb.set_trace()

if __name__ == "__main__":
    args = parse_args()
    main(args)
