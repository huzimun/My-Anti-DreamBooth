export DATASET_DIR="/data1/humw/Datasets/VGGFace2"
export EXPERIMENT_NAME="test"
export MODEL_PATH="/data1/humw/Pretrains/stable-diffusion-v1-5"
export CLASS_DIR="data/class-person"

# for person_id in `ls $DATASET_DIR`; do   
for person_id in "n000050"; do   
    export CLEAN_TRAIN_DIR=${DATASET_DIR}"/"${person_id}"/set_A" 
    export CLEAN_ADV_DIR=${DATASET_DIR}"/"${person_id}"/set_B"
    export ADV_OUTPUT_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME"/"${person_id}
    echo ${CLEAN_TRAIN_DIR}
    echo ${CLEAN_ADV_DIR}
    echo ${ADV_OUTPUT_DIR}
    # ------------------------- Train ASPL on set B -------------------------
    mkdir -p $ADV_OUTPUT_DIR
    
    # Generate Protecting Images
    accelerate launch attacks/aspl.py \
        --pretrained_model_name_or_path=$MODEL_PATH  \
        --enable_xformers_memory_efficient_attention \
        --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
        --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
        --instance_prompt="a photo of sks person" \
        --class_data_dir=$CLASS_DIR \
        --num_class_images=200 \
        --class_prompt="a photo of person" \
        --output_dir=$ADV_OUTPUT_DIR \
        --center_crop \
        --with_prior_preservation \
        --prior_loss_weight=1.0 \
        --resolution=512 \
        --train_text_encoder \
        --train_batch_size=1 \
        --max_train_steps=50 \
        --max_f_train_steps=3 \
        --max_adv_train_steps=6 \
        --checkpointing_iterations=10 \
        --learning_rate=5e-7 \
        --pgd_alpha=5e-3 \
        --pgd_eps=0.12549019607843137
done 
