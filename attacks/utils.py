import torch
import torch.nn.functional as F

prompt_dataset = [
    "Portrait of an astronaut in space, detailed starry background, reflective helmet,",
    "Painting of a floating island with giant clock gears, populated with mythical creatures,",
    "Landscape of a Japanese garden in autumn, with a bridge over a koi pond,",
    "Painting representing the sound of jazz music, using vibrant colors and erratic shapes,",
    "Painting of a modern smartphone with classic art pieces appearing on the screen,",
    "Battle scene with futuristic robots and a golden palace in the background,",
    "Scene of a bustling city market with different perspectives of people and stalls,",
    "Scene of a ship sailing in a stormy sea, with dramatic lighting and powerful waves,",
    "Portraint of a female botanist surrounded by exotic plants in a greenhouse,",
    "Painting of an ancient castle at night, with a full moon, gargoyles, and shadows,",
]

style_dataset = [
    "Art Nouveau",
    "Romantic",
    "Cubist",
    "Baroque",
    "Pop Art",
    "Abstract",
    "Impressionist",
    "Surrealist",
    "Renaissance",
    "Pointillism",
]



class attack_mixin:
    def __call__(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        unet: torch.nn.Module,
        target_tensor: torch.Tensor,
        noise_scheduler
    ):
        raise NotImplementedError
    
class AdvDM(attack_mixin):
    """
    This attack aims to maximize the training loss of diffusion model
    """
    def __call__(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        unet: torch.nn.Module,
        text_encoder: torch.nn.Module,
        input_ids,
        target_tensor: torch.Tensor,
        noise_scheduler
    ):
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

        unet.zero_grad()
        text_encoder.zero_grad()
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # target-shift loss
        if target_tensor is not None:
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

        return loss
    
class LatentAttack(attack_mixin):
    """
    This attack aims to minimize the l2 distance between latent and target_tensor
    """
    def __call__(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor=None,
        encoder_hidden_states: torch.Tensor=None,
        unet: torch.nn.Module=None,
        target_tensor: torch.Tensor=None,
        noise_scheduler=None
    ):
        if target_tensor == None:
            raise ValueError("Need a target tensor for pre-attack")
        loss = - F.mse_loss(latents, target_tensor, reduction="mean")
        return loss

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

def unet_gen(perturbed_images, trans_512, unet, vae, noise_scheduler, 
             device, weight_dtype, timesteps, encoder_hidden_states):
    perturbed_images.requires_grad = False
    if trans_512 is not None:
        latents = vae.encode(trans_512(perturbed_images).to(device, dtype=weight_dtype)).latent_dist.sample()
    else:
        latents = vae.encode(perturbed_images.to(device, dtype=weight_dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor # torch.Size([4, 4, 64, 64])

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) # torch.Size([4, 4, 64, 64])

    # Predict the noise residual
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample # torch.Size([4, 4, 64, 64])
    noisy_latents.to("cpu")
    del noisy_latents
    noise.to("cpu")
    del noise
    latents.to("cpu")
    del latents
    return model_pred

def vae_gen(perturbed_images, trans_512, vae, device, weight_dtype):
    perturbed_images.requires_grad = False
    if trans_512 is not None:
        latents = vae.encode(trans_512(perturbed_images).to(device, dtype=weight_dtype)).latent_dist.sample()
    else:
        latents = vae.encode(perturbed_images.to(device, dtype=weight_dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor # torch.Size([4, 4, 64, 64])
    return latents

def ipadapter_gen(perturbed_images, trans_224, ipadapter, device, weight_dtype):
    perturbed_images.requires_grad = False
    ipadapter_perturbed_tensor = ipadapter(trans_224(perturbed_images).to(device, 
                        dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
    return ipadapter_perturbed_tensor

def agm(args, ori_data, cur_adv, grads_dict, targets_dict, beta, models_dict, timesteps, 
    trans_512, trans_224, noise_scheduler, device, weight_dtype, encoder_hidden_states, model_types):
    """
    Adaptive gradient modulation
    :param ori_data: natural images
    :param cur_adv: adv examples in last iteration
    :param grad: gradient in this iteration
    :param label: ground truth
    :return: coefficient of each model
    """
    
    # generate adversarial example
    ori_data = ori_data.detach().clone()
    ori_data.requires_grad = False
    cur_adv = cur_adv.detach().clone()
    cur_adv.requires_grad = False
    adv_exp_dict = {}
    loss_self_dict = {}
    # import pdb; pdb.set_trace()
    for model_type in model_types:
        # adv_exp = [get_adv_example(args, ori_data=ori_data, adv_data=cur_adv, grad=grad[idx]) for idx in range(len(models))]
        # loss_self = [loss_unet, loss_vae, loss_ipadapter]
        adv_exp = get_adv_example(args, ori_data=ori_data, adv_data=cur_adv, grad=grads_dict[model_type])
        adv_exp_dict[model_type] = adv_exp
        if model_type == 'unet':
            model_pred = unet_gen(adv_exp_dict[model_type], trans_512, models_dict["unet"], models_dict["vae"], noise_scheduler, device, 
                                    weight_dtype, timesteps, encoder_hidden_states)
            if args.target_image_path == "none": # 无目标
                if args.distance_choice == "cosine":
                    loss_unet = - F.cosine_similarity(model_pred.to(weight_dtype), targets_dict["unet"].to(weight_dtype), -1).mean().item()
                else: # mse
                    loss_unet = F.mse_loss(model_pred.to(weight_dtype), targets_dict["unet"].to(weight_dtype)).item()
            else:
                if args.distance_choice == "cosine":
                    loss_unet = F.cosine_similarity(model_pred.to(weight_dtype), targets_dict["unet"].to(weight_dtype), -1).mean().item()
                else: # mse
                    loss_unet = - F.mse_loss(model_pred.to(weight_dtype), targets_dict["unet"].to(weight_dtype)).item()
            loss_self_dict[model_type] = loss_unet
            model_pred.to("cpu")
            del model_pred
        if model_type == "vae":
            latents = vae_gen(adv_exp_dict[model_type], trans_512, models_dict["vae"], device, weight_dtype)
            if args.target_image_path == "none": # 无目标
                if args.distance_choice == "cosine":
                    loss_vae = - F.cosine_similarity(latents.to(weight_dtype), targets_dict["vae"].to(weight_dtype), -1).mean().item()
                else:
                    loss_vae = F.mse_loss(latents.to(weight_dtype), targets_dict["vae"].to(weight_dtype), reduction="mean").item()
            else:
                if args.distance_choice == "cosine":
                    loss_vae = F.cosine_similarity(latents.to(weight_dtype), targets_dict["vae"].to(weight_dtype), -1).mean().item()
                else:
                    loss_vae = - F.mse_loss(latents.to(weight_dtype), targets_dict["vae"].to(weight_dtype), reduction="mean").item()
            loss_self_dict[model_type] = loss_vae
            latents.to("cpu")
            del latents
        if model_type == "ipadapter":
            ipadapter_perturbed_tensor = ipadapter_gen(adv_exp_dict[model_type], trans_224, models_dict["ipadapter"], device, weight_dtype)
            if args.target_image_path == "none": # 无目标
                if args.distance_choice == "cosine":
                    loss_ipadapter = - F.cosine_similarity(ipadapter_perturbed_tensor.to(weight_dtype), targets_dict["ipadapter"].to(weight_dtype), -1).mean().item() # -min=max
                else:
                    loss_ipadapter = F.mse_loss(ipadapter_perturbed_tensor.to(weight_dtype), targets_dict["ipadapter"].to(weight_dtype)).item()
            else:
                if args.distance_choice == "cosine":
                    loss_ipadapter = F.cosine_similarity(ipadapter_perturbed_tensor.to(weight_dtype), targets_dict["ipadapter"].to(weight_dtype), -1).mean().item() # -min=max
                else:
                    loss_ipadapter = - F.mse_loss(ipadapter_perturbed_tensor.to(weight_dtype), targets_dict["ipadapter"].to(weight_dtype)).item()
            loss_self_dict[model_type] = loss_ipadapter
            ipadapter_perturbed_tensor.to("cpu")
            del ipadapter_perturbed_tensor
    # pdb.set_trace()
    w = torch.zeros(size=(len(model_types),), device=models_dict[model_types[0]].device)

    # TODO: 支持mse损失
    # import pdb; pdb.set_trace()
    for j, model_type_j in enumerate(model_types):
        for i, model_type_i in enumerate(model_types):
            if i == j:
                continue
            elif model_type_i == "unet": # unet
                model_pred = unet_gen(adv_exp_dict[model_type_j], trans_512, models_dict["unet"], models_dict["vae"], noise_scheduler, device, 
                            weight_dtype, timesteps, encoder_hidden_states)
                if args.target_image_path == "none": # 无目标
                    if args.distance_choice == "cosine":
                        loss = - F.cosine_similarity(model_pred.to(weight_dtype), targets_dict["unet"].to(weight_dtype), -1).mean().item()
                    else:
                        loss = F.mse_loss(model_pred.to(weight_dtype), targets_dict["unet"].to(weight_dtype)).item()
                else:
                    if args.distance_choice == "cosine":
                        loss = F.cosine_similarity(model_pred.to(weight_dtype), targets_dict["unet"].to(weight_dtype), -1).mean().item()
                    else:
                        loss = - F.mse_loss(model_pred.to(weight_dtype), targets_dict["unet"].to(weight_dtype)).item()
                model_pred.to("cpu")
                del model_pred
            elif model_type_i == "vae": # vae
                latents = vae_gen(adv_exp_dict[model_type_j], trans_512, models_dict["vae"], device, weight_dtype)
                if args.target_image_path == "none": # 无目标
                    if args.distance_choice == "cosine":
                        loss = - F.cosine_similarity(latents.to(weight_dtype), targets_dict["vae"].to(weight_dtype), -1).mean().item()
                    else:
                        loss = F.mse_loss(latents.to(weight_dtype), targets_dict["vae"].to(weight_dtype), reduction="mean").item()
                else:
                    if args.distance_choice == "cosine":
                        loss = F.cosine_similarity(latents.to(weight_dtype), targets_dict["vae"].to(weight_dtype), -1).mean().item()
                    else:
                        loss = - F.mse_loss(latents.to(weight_dtype), targets_dict["vae"].to(weight_dtype), reduction="mean").item()
                latents.to("cpu")
                del latents
            elif model_type_i == "ipadapter": # ipadapter
                ipadapter_perturbed_tensor = ipadapter_gen(adv_exp_dict[model_type_j], trans_224, models_dict["ipadapter"], device, weight_dtype)
                if args.target_image_path == "none": # 无目标
                    if args.distance_choice == "cosine":
                        loss = - F.cosine_similarity(ipadapter_perturbed_tensor.to(weight_dtype), targets_dict["ipadapter"].to(weight_dtype), -1).mean().item()
                    else:
                        loss = F.mse_loss(ipadapter_perturbed_tensor.to(weight_dtype), targets_dict["ipadapter"].to(weight_dtype)).item()
                else:
                    if args.distance_choice == "cosine":
                        loss = F.cosine_similarity(ipadapter_perturbed_tensor.to(weight_dtype), targets_dict["ipadapter"].to(weight_dtype), -1).mean().item()
                    else:
                        loss = - F.mse_loss(ipadapter_perturbed_tensor.to(weight_dtype), targets_dict["ipadapter"].to(weight_dtype)).item()
                ipadapter_perturbed_tensor.to("cpu")
                del ipadapter_perturbed_tensor
            else:
                raise Exception('Wrong model index')
            w[j] += loss / loss_self_dict[model_type_i] * beta # 模型i的对抗样本扰动模型j的损失
    w = torch.softmax(w, dim=0) # 根据模拟的下一步损失确定当前的梯度权重
    # pdb.set_trace()
    del ori_data
    del cur_adv
    return w
