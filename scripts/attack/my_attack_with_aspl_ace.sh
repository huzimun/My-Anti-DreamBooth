export EXPERIMENT_NAME="ASPL"

export MODE="ace"
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${MODE}

export DATASET="CelebA-HQ"
if [ "$DATASET" = "VGGFace2" ]; then
    export DATASET_DIR="/data1/humw/Datasets/VGGFace2"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${DATASET}
elif [ "$DATASET" = "CelebA-HQ" ]; then
    export DATASET_DIR="/data1/humw/Datasets/CelebA-HQ"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${DATASET}
else
    echo "Invalid DATASET"
    exit 1
fi

export MODEL="SD15"
if [ "$MODEL" = "SD21" ]; then
    export MODEL_PATH="/data1/humw/Pretrains/stable-diffusion-2-1-base"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${MODEL}
elif [ "$MODEL" = "SD15" ]; then
    export MODEL_PATH="/data1/humw/Pretrains/stable-diffusion-v1-5"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${MODEL}
elif [ "$MODEL" = "SD14" ]; then
    export MODEL_PATH="/data1/humw/Pretrains/stable-diffusion-v1-4"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${MODEL}
else
    echo "Invalid MODEL"
    exit 1
fi

export CLASS_DIR="data/class-person"
export target="mist" # mist or yingbu
if [ "$target" = "mist" ]; then
    export target_image_path="/data1/humw/Codes/mist-v2/data/MIST.png"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${target}
elif [ "$target" = "yingbu" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/data/target_images/masks/yingbu.png"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${target}
else
    export target_image_path=""
fi

export save_config_dir="./outputs/config_scripts_logs/${EXPERIMENT_NAME}"
mkdir $save_config_dir
cp "./scripts/attack/my_attack_with_aspl_ace.sh" $save_config_dir

for person_id in `ls $DATASET_DIR`; do   
# for person_id in "n000050"; do   
    export CLEAN_TRAIN_DIR=${DATASET_DIR}"/"${person_id}"/set_A" 
    export CLEAN_ADV_DIR=${DATASET_DIR}"/"${person_id}"/set_B"
    export ADV_OUTPUT_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME"/"${person_id}
    echo ${CLEAN_TRAIN_DIR}
    echo ${CLEAN_ADV_DIR}
    echo ${ADV_OUTPUT_DIR}
    # ------------------------- Train ASPL on set B -------------------------
    mkdir -p $ADV_OUTPUT_DIR
    
    # Generate Protecting Images
    accelerate launch attacks/aspl_mist.py \
        --seed=1 \
        --mode $MODE \
        --target_image_path $target_image_path \
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


export EXPERIMENT_NAME="E-ASPL_CelebA-HQ"
export CLASS_DIR="data/class-person"
export DATASET_DIR="/data1/humw/Datasets/CelebA-HQ"

# ------------------------- Train E-ASPL on set B -------------------------
# pretrained sd models
sd14_path="/data1/humw/Pretrains/stable-diffusion-v1-4"
sd15_path="/data1/humw/Pretrains/stable-diffusion-v1-5"
sd21_path="/data1/humw/Pretrains/stable-diffusion-2-1-base"
ref_model_path="${sd14_path},${sd15_path},${sd21_path}"

for person_id in `ls $DATASET_DIR`; do    
  export CLEAN_TRAIN_DIR=${DATASET_DIR}"/"${person_id}"/set_A" 
  export CLEAN_ADV_DIR=${DATASET_DIR}"/"${person_id}"/set_B"
  export ADV_OUTPUT_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME"/"${person_id}
  echo ${CLEAN_TRAIN_DIR}
  echo ${CLEAN_ADV_DIR}
  echo ${ADV_OUTPUT_DIR}
  mkdir -p $ADV_OUTPUT_DIR

  accelerate launch attacks/ensemble_aspl.py \
    --seed=1 \
    --pretrained_model_name_or_path=${ref_model_path} \
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