export EXPERIMENT_NAME="ASPL"

export MODE="ace-plus-ipadapter"
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${MODE}

export DATASET="VGGFace2"
if [ "$DATASET" = "VGGFace2" ]; then
    export DATASET_DIR="/data1/humw/Datasets/VGGFace2"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${DATASET}
elif [ "$DATASET" = "mini-CelebA-HQ" ]; then
    export DATASET_DIR="/data1/humw/Datasets/mini-CelebA-HQ"
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

export model_types="unet" # 代理模型列表 "unet-vae-ipadapter" "unet-ipadapter" "ipadapter" "vae" "unet", "ipadaptersd"
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${model_types}

export CLASS_DIR="data/class-person"

export target="non-target"
if [ "$target" = "mist" ]; then
    export target_image_path="/data1/humw/Codes/mist-v2/data/MIST.png"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${target}
elif [ "$target" = "yingbu" ]; then
    export target_image_path="/data1/humw/Codes/FaceOff/target_images/yingbu/yingbu0.png"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${target}
elif [ "$target" = "non-target" ]; then
    export target_image_path="none"
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${target}
else
    echo "Invalid target"
    exit 1
fi

export agm=1 # 1使用自适应梯度变换，0不使用
if [ "$agm" = 1 ]; then
    export w1=1
    export w2=1
    export w3=1
    export w4=1
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm"
elif [ "$agm" = 0 ]; then
    export w1=0
    export w2=1
    export w3=0
    export w4=0
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_no-agm-w1-"${w1}"-w2-"${w2}"-w3-"${w3}"-w4-"${w4}
else
    echo "Invalid agm"
    exit 1
fi

# distance choice for adv attack loss
export distance_choice="cosine" # mse or cosine
if [ "$distance_choice" = "mse" ] || [ "$distance_choice" = "cosine" ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${distance_choice}
else
    echo "Invalid distance_choice"
    exit 1
fi

# use EOT
export eot=0 # 1使用EOT，0不使用
if [ "$eot" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_eot-1"
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_eot-0"
fi

export sds=0 # 1使用SDS，0不使用
if [ "$sds" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_sds-1"
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_sds-0"
fi

export untargeted_unet=1 # 1使用untargeted unet，0不使用
if [ "$untargeted_unet" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_unT-unet-1"
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_unT-unet-0"
fi
d

export fix_unet=0 # 1使用固定unet，0不使用
if [ "$fix_unet" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_fix-unet-1"
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_fix-unet-0"
fi

export fix_noise=0 # 1使用固定噪声，0不使用
if [ "$fix_noise" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_fix-noise-1"
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_fix-noise-0"
fi

# 数据增强
# if gau is selected, the gauK is the kernel size of guassian filter and hflip
# if std, just resize and center crop
# if bsr, block shuffle and rotation
export train_mode=std
export gauK=7
if [ "$train_mode" = "gau" ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_gau-"$gauK
elif [ "$train_mode" = "bsr" ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_bsr"
else
    echo "train_mode default to be std"
fi

export save_config_dir="./outputs/config_scripts_logs/${EXPERIMENT_NAME}"
mkdir $save_config_dir
cp "./scripts/attack/my_attack_with_aspl_ensemble_faceoff.sh" $save_config_dir

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
    command="""accelerate launch attacks/aspl_ensemble_faceoff.py \
        --seed=1 \
        --fix_noise $fix_noise \
        --fix_unet $fix_unet \
        --untargeted_unet $untargeted_unet \
        --sds $sds \
        --model_types $model_types \
        --agm $agm \
        --w1 $w1 \
        --w2 $w2 \
        --w3 $w3 \
        --w4 $w4 \
        --mode $MODE \
        --eot $eot \
        --distance_choice $distance_choice \
        --target_image_path $target_image_path \
        --pretrained_model_name_or_path=$MODEL_PATH  \
        --enable_xformers_memory_efficient_attention \
        --instance_data_dir_for_train=$CLEAN_TRAIN_DIR \
        --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
        --instance_prompt='a photo of sks person' \
        --class_data_dir=$CLASS_DIR \
        --num_class_images=200 \
        --class_prompt='a photo of person' \
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
        --pgd_eps=0.12549019607843137"""
    
    if [ "$train_mode" = "gau" ]; then
        command="$command --transform_gau --gau_kernel_size $gauK --transform_hflip "
    elif [ "$train_mode" = "bsr" ]; then
        command="$command --bsr"
    else
        echo "train_mode default to be std"
    fi

    echo $command
    eval $command
        
done 
