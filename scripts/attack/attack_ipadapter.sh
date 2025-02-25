export adversarial_folder_name="test_ipadapter_sd1-5"
export experiment_name="test_ipadapter_sd1-5"
export device="cuda:6"
export save_config_dir="./outputs/config_scripts_logs/${experiment_name}"
mkdir $save_config_dir
cp "./scripts/attack/attack_ipadapter.sh" $save_config_dir
python ./attacks/attack_ipadapter.py \
    --model_type "sd15" \
    --pretrained_model_name_or_path "/home/humw/Pretrains/stable-diffusion-v1-5" \
    --image_encoder_path "/home/humw/Pretrains/h94/IP-Adapter/models/image_encoder" \
    --ip_ckpt "/home/humw/Pretrains/h94/IP-Adapter/models/ip-adapter-plus_sd15.bin" \
    --vae_model_path "/home/humw/Pretrains/sd-vae-ft-mse" \
    --device $device \
    --input_dir "/home/humw/Codes/My-Anti-DreamBooth/data/test" \
    --output_dir "./outputs/adversarial_images/${experiment_name}" \
    --resolution 224 \
    --sub_name "set_B" \
    --prior_generation_precision "fp16" \
    --pgd_alpha=5e-3 \
    --pgd_eps=0.12549019607843137
