export DATASET_DIR="./outputs/adversarial_images/Clean_test2"
export EXPERIMENT_NAME="Clean_test2_sd1-5_bs1_not_save_checkpoint_infer-bs1_mist-v2_2000"
export MODEL_PATH="./stable-diffusion/stable-diffusion-v1-5"
export CLASS_DIR="data/class-person"

for person_id in `ls $DATASET_DIR`; do       
    # ------------------------- Train DreamBooth on perturbed examples -------------------------
    export INSTANCE_DIR=${DATASET_DIR}"/"${person_id}
    export DREAMBOOTH_OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME"/"${person_id}
    echo ${INSTANCE_DIR}
    echo ${DREAMBOOTH_OUTPUT_DIR}
    
    accelerate launch my_train_dreambooth.py \
      --pretrained_model_name_or_path=$MODEL_PATH  \
      --enable_xformers_memory_efficient_attention \
      --train_text_encoder \
      --instance_data_dir=$INSTANCE_DIR \
      --class_data_dir=$CLASS_DIR \
      --output_dir=$DREAMBOOTH_OUTPUT_DIR \
      --with_prior_preservation \
      --prior_loss_weight=1.0 \
      --instance_prompt="a photo of sks person" \
      --class_prompt="a photo of person" \
      --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
      --resolution=512 \
      --train_batch_size=1 \
      --gradient_accumulation_steps=1 \
      --learning_rate=1e-4 \
      --lr_scheduler="constant" \
      --lr_warmup_steps=0 \
      --num_class_images=200 \
      --max_train_steps=2000 \
      --checkpointing_steps=2000 \
      --center_crop \
      --mixed_precision=bf16 \
      --prior_generation_precision=bf16 \
      --sample_batch_size=8
done 


# for person_id in `ls $DATASET_DIR`; do       
#     # ------------------------- Train DreamBooth on perturbed examples -------------------------
#     export INSTANCE_DIR=${DATASET_DIR}"/"${person_id}
#     export DREAMBOOTH_OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME"/"${person_id}
#     echo ${INSTANCE_DIR}
#     echo ${DREAMBOOTH_OUTPUT_DIR}
    
#     accelerate launch my_train_dreambooth.py \
#       --pretrained_model_name_or_path=$MODEL_PATH  \
#       --enable_xformers_memory_efficient_attention \
#       --train_text_encoder \
#       --instance_data_dir=$INSTANCE_DIR \
#       --class_data_dir=$CLASS_DIR \
#       --output_dir=$DREAMBOOTH_OUTPUT_DIR \
#       --with_prior_preservation \
#       --prior_loss_weight=1.0 \
#       --instance_prompt="a photo of sks person" \
#       --class_prompt="a photo of person" \
#       --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
#       --resolution=512 \
#       --train_batch_size=1 \
#       --gradient_accumulation_steps=1 \
#       --learning_rate=5e-7 \
#       --lr_scheduler="constant" \
#       --lr_warmup_steps=0 \
#       --num_class_images=200 \
#       --max_train_steps=1000 \
#       --checkpointing_steps=1000 \
#       --center_crop \
#       --mixed_precision=bf16 \
#       --prior_generation_precision=bf16 \
#       --sample_batch_size=8
# done 
