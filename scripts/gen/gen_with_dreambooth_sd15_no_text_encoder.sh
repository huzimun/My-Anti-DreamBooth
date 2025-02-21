# export EXPERIMENT_NAME="unet_ipadapter_test_mse_w1_num200_alpha6_eps16_input512_yingbu"
# export DATASET_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME
# export OUTPUT_DIR="outputs/customization_outputs/${EXPERIMENT_NAME}no_text_encoder"
# export MODEL_PATH="stable-diffusion/stable-diffusion-v1-5"
# export CLASS_DIR="data/class-person"

# for person_id in `ls $DATASET_DIR`; do
#     # Train DreamBooth on Protecting Images
#     export INSTANCE_DIR=${DATASET_DIR}"/"${person_id}
#     export DREAMBOOTH_OUTPUT_DIR=$OUTPUT_DIR"/"${person_id}
#     echo ${INSTANCE_DIR}
#     echo ${DREAMBOOTH_OUTPUT_DIR}
    
#     accelerate launch train_dreambooth.py \
#         --pretrained_model_name_or_path=$MODEL_PATH  \
#         --enable_xformers_memory_efficient_attention \
#         --instance_data_dir=$INSTANCE_DIR \
#         --class_data_dir=$CLASS_DIR \
#         --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#         --with_prior_preservation \
#         --prior_loss_weight=1.0 \
#         --instance_prompt="a photo of sks person" \
#         --class_prompt="a photo of person" \
#         --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
#         --resolution=512 \
#         --train_batch_size=2 \
#         --gradient_accumulation_steps=1 \
#         --learning_rate=5e-7 \
#         --lr_scheduler="constant" \
#         --lr_warmup_steps=0 \
#         --num_class_images=200 \
#         --max_train_steps=1000 \
#         --checkpointing_steps=1000 \
#         --center_crop \
#         --mixed_precision "bf16" \
#         --prior_generation_precision "bf16" \
#         --sample_batch_size=8
# done

export EXPERIMENT_NAME="VGGFace2_SD15_no_text_encoder"
export DATASET_DIR="outputs/adversarial_images/Clean_test2"
export OUTPUT_DIR="outputs/customization_outputs/VGGFace2_SD15_no_text_encoder"
export MODEL_PATH="stable-diffusion/stable-diffusion-v1-5"
export CLASS_DIR="data/class-person"

for person_id in `ls $DATASET_DIR`; do
    # Train DreamBooth on Protecting Images
    export INSTANCE_DIR=${DATASET_DIR}"/"${person_id}
    export DREAMBOOTH_OUTPUT_DIR=$OUTPUT_DIR"/"${person_id}
    echo ${INSTANCE_DIR}
    echo ${DREAMBOOTH_OUTPUT_DIR}
    
    accelerate launch train_dreambooth.py \
        --pretrained_model_name_or_path=$MODEL_PATH  \
        --enable_xformers_memory_efficient_attention \
        --instance_data_dir=$INSTANCE_DIR \
        --class_data_dir=$CLASS_DIR \
        --output_dir=$DREAMBOOTH_OUTPUT_DIR \
        --with_prior_preservation \
        --prior_loss_weight=1.0 \
        --instance_prompt="a photo of sks person" \
        --class_prompt="a photo of person" \
        --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
        --resolution=512 \
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision "bf16" \
        --prior_generation_precision "bf16" \
        --sample_batch_size=8
done
