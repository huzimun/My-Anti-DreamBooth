export MODEL_PATH="stable-diffusion/stable-diffusion-v2-1"
export INSTANCE_DIR="/gemini/code/CAAT_noise-16_mini-10"
export SUB_NAME='' #'set_B'
export CLASS_DIR="data/class-person"
export DREAMBOOTH_OUTPUT_DIR="dreambooth-outputs/CAAT_noise-16_mini-10" # 
export image_output_dir='outputs/CAAT_noise-16_mini-10'

mkdir $image_output_dir

# for person_id in `ls $INSTANCE_DIR`; do
for person_id in "n000050" "n000057" "n000063" "n000068" "n000076" "n000139" "n000146" "n000171"; do      
    person_dir=$INSTANCE_DIR'/'${person_id}'/'$SUB_NAME
    echo ${person_dir}
    accelerate launch train_dreambooth.py \
      --pretrained_model_name_or_path=$MODEL_PATH  \
      --enable_xformers_memory_efficient_attention \
      --train_text_encoder \
      --instance_data_dir=${person_dir} \
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
      --checkpointing_steps=500 \
      --center_crop \
      --mixed_precision=bf16 \
      --prior_generation_precision=bf16 \
      --sample_batch_size=8
    image_output_path=$image_output_dir'/'${person_id}
    echo ${image_output_path}
    mkdir "${image_output_path}"
    cp -r $DREAMBOOTH_OUTPUT_DIR'/checkpoint-1000/dreambooth/a_dslr_portrait_of_sks_person' $image_output_path   
    cp -r $DREAMBOOTH_OUTPUT_DIR'/checkpoint-1000/dreambooth/a_photo_of_sks_person' $image_output_path                             
done  