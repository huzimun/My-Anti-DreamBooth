export EXPERIMENT_NAME="vae_VGGFace2_w1.0_num100_alpha6_eps16_input512_512_yingbu_refiner0"
export DATASET_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME
export OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME
export MODEL_PATH="stable-diffusion/stable-diffusion-2-1-base"
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
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=8
done 

export EXPERIMENT_NAME="vae_VGGFace2_w0.75_num100_alpha6_eps16_input512_512_yingbu_refiner0"
export DATASET_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME
export OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME
export MODEL_PATH="stable-diffusion/stable-diffusion-2-1-base"
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
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=8
done 

export EXPERIMENT_NAME="vae_VGGFace2_w0.5_num100_alpha6_eps16_input512_512_yingbu_refiner0"
export DATASET_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME
export OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME
export MODEL_PATH="stable-diffusion/stable-diffusion-2-1-base"
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
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=8
done 

export EXPERIMENT_NAME="vae_VGGFace2_w0.25_num100_alpha6_eps16_input512_512_yingbu_refiner0"
export DATASET_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME
export OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME
export MODEL_PATH="stable-diffusion/stable-diffusion-2-1-base"
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
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=8
done 

export EXPERIMENT_NAME="vae_VGGFace2_w0.0_num100_alpha6_eps16_input512_512_yingbu_refiner0"
export DATASET_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME
export OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME
export MODEL_PATH="stable-diffusion/stable-diffusion-2-1-base"
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
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=8
done 

export EXPERIMENT_NAME="vae_VGGFace2_w0.0_num100_alpha6_eps16_input512_512_mist_refiner0"
export DATASET_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME
export OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME
export MODEL_PATH="stable-diffusion/stable-diffusion-2-1-base"
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
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=8
done 

export EXPERIMENT_NAME="vae_VGGFace2_w1.0_num100_alpha6_eps8_input512_512_yingbu_refiner0"
export DATASET_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME
export OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME
export MODEL_PATH="stable-diffusion/stable-diffusion-2-1-base"
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
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=8
done 

export EXPERIMENT_NAME="vae_VGGFace2_w1.0_num100_alpha6_eps12_input512_512_yingbu_refiner0"
export DATASET_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME
export OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME
export MODEL_PATH="stable-diffusion/stable-diffusion-2-1-base"
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
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=8
done 

export EXPERIMENT_NAME="vae_VGGFace2_w1.0_num100_alpha6_eps16_input512_512_yingbu_refiner1_edge200-100_filter3_min-eps8_interval10"
export DATASET_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME
export OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME
export MODEL_PATH="stable-diffusion/stable-diffusion-2-1-base"
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
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=8
done 

export EXPERIMENT_NAME="vae_VGGFace2_w1.0_num100_alpha6_eps16_input512_512_yingbu_refiner1_edge200-100_filter3_min-eps12_interval10"
export DATASET_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME
export OUTPUT_DIR="outputs/customization_outputs/"$EXPERIMENT_NAME
export MODEL_PATH="stable-diffusion/stable-diffusion-2-1-base"
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
        --train_batch_size=2 \
        --gradient_accumulation_steps=1 \
        --learning_rate=5e-7 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --num_class_images=200 \
        --max_train_steps=1000 \
        --checkpointing_steps=1000 \
        --center_crop \
        --mixed_precision=bf16 \
        --prior_generation_precision=bf16 \
        --sample_batch_size=8
done
