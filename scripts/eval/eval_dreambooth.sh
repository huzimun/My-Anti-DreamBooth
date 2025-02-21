export adversarial_folder_name="vae_VGGFace2_mse_w0.0_num100_alpha6_eps16_input512_512_max_refiner0"
echo $adversarial_folder_name
export device="cuda:2"
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export original_output_dir="./outputs/customization_outputs/VGGFace2"
export prompts="a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export VGGFace2="/data1/humw/Datasets/VGGFace2"
export clip_model_name_or_path="/data1/humw/Codes/Anti-DreamBooth/evaluations/my_clip/ViT-B-32.pt"
echo $prompts

# # IMS: protected output and original input
# # ArcFace
# python ./evaluations/ism_fdfr.py \
#     --prompts $prompts \
#     --data_dir $customization_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "protected_output" \
#     --scene2 "original_input" \
#     --is_target 0 \
#     --map_path "" \
#     --target_path "" \
#     --model_name "ArcFace" \
#     --input_name "set_B" \
#     --out_out 0
# # VGG-Face
# python ./evaluations/ism_fdfr.py \
#     --prompts $prompts \
#     --data_dir $customization_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "protected_output" \
#     --scene2 "original_input" \
#     --is_target 0 \
#     --map_path "" \
#     --target_path "" \
#     --model_name "VGG-Face" \
#     --input_name "set_B" \
#     --out_out 0
# # CLIP
# python ./evaluations/my_clip/my_clip.py \
#     --prompts $prompts \
#     --data_dir $customization_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "protected_output" \
#     --scene2 "original_input" \
#     --is_target 0 \
#     --map_path "" \
#     --target_path "" \
#     --model_name_or_path $clip_model_name_or_path \
#     --device $device \
#     --input_name "" \
#     --out_out 0

# # original output and original input
# # ArcFace
# python ./evaluations/ism_fdfr.py \
#     --prompts $prompts \
#     --data_dir $original_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "original_output" \
#     --scene2 "original_input" \
#     --is_target 0 \
#     --map_path "" \
#     --target_path "" \
#     --model_name "ArcFace" \
#     --input_name "set_B" \
#     --out_out 0
# # VGG-Face
# python ./evaluations/ism_fdfr.py \
#     --prompts $prompts \
#     --data_dir $original_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "original_output" \
#     --scene2 "original_input" \
#     --is_target 0 \
#     --map_path "" \
#     --target_path "" \
#     --model_name "VGG-Face" \
#     --input_name "set_B" \
#     --out_out 0
# # CLIP
# python ./evaluations/my_clip/my_clip.py \
#     --prompts $prompts \
#     --data_dir $original_output_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "original_output" \
#     --scene2 "original_input" \
#     --is_target 0 \
#     --map_path "" \
#     --target_path "" \
#     --model_name_or_path $clip_model_name_or_path \
#     --device $device \
#     --input_name "" \
#     --out_out 0

# # image quality evaluation
# # LIQE: protected input
# python ./evaluations/LIQE/run_liqe_for_input.py \
#     --data_dir $adversarial_input_dir \
#     --sub_folder "" \
#     --save_dir $evaluation_output_dir \
#     --scene "protected_input" \
#     --device $device
python ./evaluations/LIQE/run_liqe_for_input.py \
    --data_dir $VGGFace2 \
    --sub_folder "set_B" \
    --save_dir $evaluation_output_dir \
    --scene "original_input" \
    --device $device
# # lpips: protected_input and original_input
# python ./evaluations/lpips/my_lpips.py \
#     --data_dir $adversarial_input_dir \
#     --emb_dirs $VGGFace2 \
#     --save_dir $evaluation_output_dir \
#     --scene "protected_input" \
#     --scene2 "original_input" \
#     --model_name_or_path "alex" \
#     --device $device \
#     --resolution 512
# # BRISQUE: protected input
# # FID: protected_input and original_input
# # SSIM
# # PSNR
# # LDM