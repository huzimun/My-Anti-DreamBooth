export adversarial_folder_name="ensemble_sds5_eps16_steps100_gmode+"
echo $adversarial_folder_name
export device="cuda:0"
export adversarial_input_dir="./outputs/adversarial_images/${adversarial_folder_name}"
export customization_output_dir="./outputs/customization_outputs/${adversarial_folder_name}"
export evaluation_output_dir="./outputs/evaluation_outputs/${adversarial_folder_name}"
export original_output_dir="./outputs/customization_outputs/VGGFace2"
export prompts="a_photo_of_sks_person;a_dslr_portrait_of_sks_person"
export VGGFace2="/data1/humw/Datasets/VGGFace2"
export clip_model_name_or_path="/data1/humw/Codes/Anti-DreamBooth/evaluations/my_clip/ViT-B-32.pt"
echo $prompts

# IMS: protected output and original input
# ArcFace
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path "" \
    --target_path "" \
    --model_name "ArcFace" \
    --input_name "set_B" \
    --out_out 0
# VGG-Face
python ./evaluations/ism_fdfr.py \
    --prompts $prompts \
    --data_dir $customization_output_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --is_target 0 \
    --map_path "" \
    --target_path "" \
    --model_name "VGG-Face" \
    --input_name "set_B" \
    --out_out 0

# IQA: protected output and original input
# FID
# LIQE
# BRISQUE
python ./evaluations/pyiqa/iqa_metric_for_output.py \
    --data_dir $customization_output_dir \
    --emb_dir $VGGFace2 \
    --prompts $prompts \
    --save_dir $evaluation_output_dir \
    --scene "protected_output" \
    --scene2 "original_input" \
    --device $device

# protected_input and original_input: FID, LPIPS, SSIM, PSNR
# protected_input: LIQE, BRISQUE
python ./evaluations/pyiqa/iqa_metric.py \
    --data_dir $adversarial_input_dir \
    --emb_dirs $VGGFace2 \
    --save_dir $evaluation_output_dir \
    --sub_folder '' \
    --scene "protected_input" \
    --scene2 "original_input" \
    --device $device
