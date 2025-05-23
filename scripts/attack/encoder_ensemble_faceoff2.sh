

export EXPERIMENT_NAME="Encoder_attack_conda-photomaker"
export device="cuda:3"

export DATASET="VGGFace2"
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${DATASET}
if [ "$DATASET" = "VGGFace2" ]; then
    export DATASET_DIR="/data1/humw/Datasets/VGGFace2"
elif [ "$DATASET" = "CelebA-HQ" ]; then
    export DATASET_DIR="/data1/humw/Datasets/CelebA-HQ"
elif [ "$DATASET" = "Tongyi_CelebA" ]; then
    export DATASET_DIR="/data1/humw/Datasets/Tongyi_CelebA"
elif [ "$DATASET" = "New_Tongyi_CelebA" ]; then
    export DATASET_DIR="/data1/humw/Datasets/New_Tongyi_CelebA"
elif [ "$DATASET" = "mini-CelebA-HQ_sr_no128" ]; then
    export DATASET_DIR="/data1/humw/Datasets/New_Tongyi_CelebA"
else
    echo "Invalid DATASET"
    exit 1
fi
# "vae14-vae15-vae21-ipadapter-photomaker-pulid"
export model_types="vae15-ipadapter-photomaker"
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${model_types}

# distance choice for adv attack loss
export distance_choice="mix" # mse or cosine, mix use mse for vae, and cosine for ipadapter and photomaker
if [ "$distance_choice" = "mse" ] || [ "$distance_choice" = "cosine" ] || [ "$distance_choice" = "mix" ]; then
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

export target="chengyaojin" # "yingbu" "mist" "max-mask" "min-mask" "random-mask"
export id_map_path="test" # map original id to target id
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${target}
if [ "$target" = "mist" ]; then
    export target_image_path="/data1/humw/Codes/FaceOff/target_images/mist"
elif [ "$target" = "yingbu" ]; then
    export target_image_path="/data1/humw/Codes/FaceOff/target_images/yingbu"
elif [ "$target" = "baozheng" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/baozheng"
elif [ "$target" = "caocao" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/caocao"
elif [ "$target" = "chengyaojin" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/chengyaojin"
elif [ "$target" = "guanyu" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/guanyu"
elif [ "$target" = "lumeng" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/lumeng"
elif [ "$target" = "max-mask" ] || [ "$target" = "min-mask" ] || [ "$target" = "random-mask" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/target_images"
    if [ "$target" = "max-mask" ]; then
        if [ "$DATASET" = "CelebA-HQ" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/max_vae_mse_CelebA-HQ-mask.json"
        elif [ "$DATASET" = "VGGFace2" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/max_vae_mse_VGGFace2-mask.json"
        else
            export "invalid DATASET"
            exit 1
        fi
    elif [ "$target" = "min-mask" ]; then
        if [ "$DATASET" = "CelebA-HQ" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/min_vae_mse_CelebA-HQ-mask.json"
        elif [ "$DATASET" = "VGGFace2" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/min_vae_mse_VGGFace2-mask.json"
        else
            export "invalid DATASET"
            exit 1
        fi
    elif [ "$target" = "random-mask" ]; then
        if [ "$DATASET" = "CelebA-HQ" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/random_vae_mse_CelebA-HQ-mask.json"
        elif [ "$DATASET" = "VGGFace2" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/random_vae_mse_VGGFace2-mask.json"
        else
            export "invalid DATASET"
            exit 1
        fi
    fi
elif [ "$target" = "non-target" ]; then
    export target_image_path="non-target"
else
    echo "Invalid target"
    exit 1
fi

# use AGM
export agm=2 # 1使用AGM，0取平均，2手动调整10:1:1，3手动调整5：1：1
if [ "$agm" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-1"
elif [ "$agm" = 0 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-0"
elif [ "$agm" = 2 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-2"
elif [ "$agm" = 3 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-3"
else
    export "invalid agm"
    exit 1
fi

# use normalization
export norm=0 # 1使用AGM，0不使用
if [ "$norm" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_norm-1"
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_norm-0"
fi

export save_config_dir="./outputs/config_scripts_logs/${EXPERIMENT_NAME}"
mkdir $save_config_dir
cp "./scripts/attack/encoder_ensemble_faceoff2.sh" $save_config_dir

# for person_id in `ls $DATASET_DIR`; do   
for person_id in "n000215" "n000217" "n000220" "n000221" "n000223" "n000225" "n000228" "n000234" "n000236" "n000238" "n000243"; do   
    export CLEAN_ADV_DIR=${DATASET_DIR}"/"${person_id}"/set_B"
    export ADV_OUTPUT_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME"/"${person_id}
    echo ${CLEAN_ADV_DIR}
    echo ${ADV_OUTPUT_DIR}
    # ------------------------- Train ASPL on set B -------------------------
    mkdir -p $ADV_OUTPUT_DIR
    
    # Generate Protecting Images
    python3 attacks/encoder_ensemble_faceoff.py \
        --id_map_path $id_map_path \
        --norm $norm \
        --agm $agm \
        --model_types $model_types \
        --device=$device \
        --seed=1 \
        --eot $eot \
        --target $target \
        --distance_choice $distance_choice \
        --target_image_path $target_image_path \
        --pretrained_model_name_or_path=$MODEL_PATH  \
        --mixed_precision "bf16" \
        --enable_xformers_memory_efficient_attention \
        --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
        --output_dir=$ADV_OUTPUT_DIR \
        --center_crop \
        --resolution=512 \
        --max_train_steps=50 \
        --max_adv_train_steps=6 \
        --pgd_alpha=5e-3 \
        --pgd_eps=0.12549019607843137
        
done 



export EXPERIMENT_NAME="Encoder_attack_conda-photomaker"
export device="cuda:3"

export DATASET="VGGFace2"
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${DATASET}
if [ "$DATASET" = "VGGFace2" ]; then
    export DATASET_DIR="/data1/humw/Datasets/VGGFace2"
elif [ "$DATASET" = "CelebA-HQ" ]; then
    export DATASET_DIR="/data1/humw/Datasets/CelebA-HQ"
elif [ "$DATASET" = "Tongyi_CelebA" ]; then
    export DATASET_DIR="/data1/humw/Datasets/Tongyi_CelebA"
elif [ "$DATASET" = "New_Tongyi_CelebA" ]; then
    export DATASET_DIR="/data1/humw/Datasets/New_Tongyi_CelebA"
elif [ "$DATASET" = "mini-CelebA-HQ_sr_no128" ]; then
    export DATASET_DIR="/data1/humw/Datasets/New_Tongyi_CelebA"
else
    echo "Invalid DATASET"
    exit 1
fi
# "vae14-vae15-vae21-ipadapter-photomaker-pulid"
export model_types="vae15-ipadapter-photomaker"
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${model_types}

# distance choice for adv attack loss
export distance_choice="mix" # mse or cosine, mix use mse for vae, and cosine for ipadapter and photomaker
if [ "$distance_choice" = "mse" ] || [ "$distance_choice" = "cosine" ] || [ "$distance_choice" = "mix" ]; then
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

export target="guanyu" # "yingbu" "mist" "max-mask" "min-mask" "random-mask"
export id_map_path="test" # map original id to target id
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${target}
if [ "$target" = "mist" ]; then
    export target_image_path="/data1/humw/Codes/FaceOff/target_images/mist"
elif [ "$target" = "yingbu" ]; then
    export target_image_path="/data1/humw/Codes/FaceOff/target_images/yingbu"
elif [ "$target" = "baozheng" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/baozheng"
elif [ "$target" = "caocao" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/caocao"
elif [ "$target" = "chengyaojin" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/chengyaojin"
elif [ "$target" = "guanyu" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/guanyu"
elif [ "$target" = "lumeng" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/lumeng"
elif [ "$target" = "max-mask" ] || [ "$target" = "min-mask" ] || [ "$target" = "random-mask" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/target_images"
    if [ "$target" = "max-mask" ]; then
        if [ "$DATASET" = "CelebA-HQ" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/max_vae_mse_CelebA-HQ-mask.json"
        elif [ "$DATASET" = "VGGFace2" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/max_vae_mse_VGGFace2-mask.json"
        else
            export "invalid DATASET"
            exit 1
        fi
    elif [ "$target" = "min-mask" ]; then
        if [ "$DATASET" = "CelebA-HQ" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/min_vae_mse_CelebA-HQ-mask.json"
        elif [ "$DATASET" = "VGGFace2" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/min_vae_mse_VGGFace2-mask.json"
        else
            export "invalid DATASET"
            exit 1
        fi
    elif [ "$target" = "random-mask" ]; then
        if [ "$DATASET" = "CelebA-HQ" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/random_vae_mse_CelebA-HQ-mask.json"
        elif [ "$DATASET" = "VGGFace2" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/random_vae_mse_VGGFace2-mask.json"
        else
            export "invalid DATASET"
            exit 1
        fi
    fi
elif [ "$target" = "non-target" ]; then
    export target_image_path="non-target"
else
    echo "Invalid target"
    exit 1
fi

# use AGM
export agm=2 # 1使用AGM，0取平均，2手动调整10:1:1，3手动调整5：1：1
if [ "$agm" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-1"
elif [ "$agm" = 0 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-0"
elif [ "$agm" = 2 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-2"
elif [ "$agm" = 3 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-3"
else
    export "invalid agm"
    exit 1
fi

# use normalization
export norm=0 # 1使用AGM，0不使用
if [ "$norm" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_norm-1"
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_norm-0"
fi

export save_config_dir="./outputs/config_scripts_logs/${EXPERIMENT_NAME}"
mkdir $save_config_dir
cp "./scripts/attack/encoder_ensemble_faceoff2.sh" $save_config_dir

for person_id in `ls $DATASET_DIR`; do   
# for person_id in "n000050" n000057; do   
    export CLEAN_ADV_DIR=${DATASET_DIR}"/"${person_id}"/set_B"
    export ADV_OUTPUT_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME"/"${person_id}
    echo ${CLEAN_ADV_DIR}
    echo ${ADV_OUTPUT_DIR}
    # ------------------------- Train ASPL on set B -------------------------
    mkdir -p $ADV_OUTPUT_DIR
    
    # Generate Protecting Images
    python3 attacks/encoder_ensemble_faceoff.py \
        --id_map_path $id_map_path \
        --norm $norm \
        --agm $agm \
        --model_types $model_types \
        --device=$device \
        --seed=1 \
        --eot $eot \
        --target $target \
        --distance_choice $distance_choice \
        --target_image_path $target_image_path \
        --pretrained_model_name_or_path=$MODEL_PATH  \
        --mixed_precision "bf16" \
        --enable_xformers_memory_efficient_attention \
        --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
        --output_dir=$ADV_OUTPUT_DIR \
        --center_crop \
        --resolution=512 \
        --max_train_steps=50 \
        --max_adv_train_steps=6 \
        --pgd_alpha=5e-3 \
        --pgd_eps=0.12549019607843137
        
done 



export EXPERIMENT_NAME="Encoder_attack_conda-photomaker"
export device="cuda:3"

export DATASET="VGGFace2"
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${DATASET}
if [ "$DATASET" = "VGGFace2" ]; then
    export DATASET_DIR="/data1/humw/Datasets/VGGFace2"
elif [ "$DATASET" = "CelebA-HQ" ]; then
    export DATASET_DIR="/data1/humw/Datasets/CelebA-HQ"
elif [ "$DATASET" = "Tongyi_CelebA" ]; then
    export DATASET_DIR="/data1/humw/Datasets/Tongyi_CelebA"
elif [ "$DATASET" = "New_Tongyi_CelebA" ]; then
    export DATASET_DIR="/data1/humw/Datasets/New_Tongyi_CelebA"
elif [ "$DATASET" = "mini-CelebA-HQ_sr_no128" ]; then
    export DATASET_DIR="/data1/humw/Datasets/New_Tongyi_CelebA"
else
    echo "Invalid DATASET"
    exit 1
fi
# "vae14-vae15-vae21-ipadapter-photomaker-pulid"
export model_types="vae15-ipadapter-photomaker"
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${model_types}

# distance choice for adv attack loss
export distance_choice="mix" # mse or cosine, mix use mse for vae, and cosine for ipadapter and photomaker
if [ "$distance_choice" = "mse" ] || [ "$distance_choice" = "cosine" ] || [ "$distance_choice" = "mix" ]; then
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

export target="lumeng" # "yingbu" "mist" "max-mask" "min-mask" "random-mask"
export id_map_path="test" # map original id to target id
EXPERIMENT_NAME=${EXPERIMENT_NAME}"_"${target}
if [ "$target" = "mist" ]; then
    export target_image_path="/data1/humw/Codes/FaceOff/target_images/mist"
elif [ "$target" = "yingbu" ]; then
    export target_image_path="/data1/humw/Codes/FaceOff/target_images/yingbu"
elif [ "$target" = "baozheng" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/baozheng"
elif [ "$target" = "caocao" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/caocao"
elif [ "$target" = "chengyaojin" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/chengyaojin"
elif [ "$target" = "guanyu" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/guanyu"
elif [ "$target" = "lumeng" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/mask_targets/lumeng"
elif [ "$target" = "max-mask" ] || [ "$target" = "min-mask" ] || [ "$target" = "random-mask" ]; then
    export target_image_path="/data1/humw/Codes/My-Anti-DreamBooth/target_images"
    if [ "$target" = "max-mask" ]; then
        if [ "$DATASET" = "CelebA-HQ" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/max_vae_mse_CelebA-HQ-mask.json"
        elif [ "$DATASET" = "VGGFace2" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/max_vae_mse_VGGFace2-mask.json"
        else
            export "invalid DATASET"
            exit 1
        fi
    elif [ "$target" = "min-mask" ]; then
        if [ "$DATASET" = "CelebA-HQ" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/min_vae_mse_CelebA-HQ-mask.json"
        elif [ "$DATASET" = "VGGFace2" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/min_vae_mse_VGGFace2-mask.json"
        else
            export "invalid DATASET"
            exit 1
        fi
    elif [ "$target" = "random-mask" ]; then
        if [ "$DATASET" = "CelebA-HQ" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/random_vae_mse_CelebA-HQ-mask.json"
        elif [ "$DATASET" = "VGGFace2" ]; then
            export id_map_path="/data1/humw/Codes/My-Anti-DreamBooth/random_vae_mse_VGGFace2-mask.json"
        else
            export "invalid DATASET"
            exit 1
        fi
    fi
elif [ "$target" = "non-target" ]; then
    export target_image_path="non-target"
else
    echo "Invalid target"
    exit 1
fi

# use AGM
export agm=2 # 1使用AGM，0取平均，2手动调整10:1:1，3手动调整5：1：1
if [ "$agm" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-1"
elif [ "$agm" = 0 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-0"
elif [ "$agm" = 2 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-2"
elif [ "$agm" = 3 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_agm-3"
else
    export "invalid agm"
    exit 1
fi

# use normalization
export norm=0 # 1使用AGM，0不使用
if [ "$norm" = 1 ]; then
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_norm-1"
else
    EXPERIMENT_NAME=${EXPERIMENT_NAME}"_norm-0"
fi

export save_config_dir="./outputs/config_scripts_logs/${EXPERIMENT_NAME}"
mkdir $save_config_dir
cp "./scripts/attack/encoder_ensemble_faceoff2.sh" $save_config_dir

for person_id in `ls $DATASET_DIR`; do   
# for person_id in "n000050" n000057; do   
    export CLEAN_ADV_DIR=${DATASET_DIR}"/"${person_id}"/set_B"
    export ADV_OUTPUT_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME"/"${person_id}
    echo ${CLEAN_ADV_DIR}
    echo ${ADV_OUTPUT_DIR}
    # ------------------------- Train ASPL on set B -------------------------
    mkdir -p $ADV_OUTPUT_DIR
    
    # Generate Protecting Images
    python3 attacks/encoder_ensemble_faceoff.py \
        --id_map_path $id_map_path \
        --norm $norm \
        --agm $agm \
        --model_types $model_types \
        --device=$device \
        --seed=1 \
        --eot $eot \
        --target $target \
        --distance_choice $distance_choice \
        --target_image_path $target_image_path \
        --pretrained_model_name_or_path=$MODEL_PATH  \
        --mixed_precision "bf16" \
        --enable_xformers_memory_efficient_attention \
        --instance_data_dir_for_adversarial=$CLEAN_ADV_DIR \
        --output_dir=$ADV_OUTPUT_DIR \
        --center_crop \
        --resolution=512 \
        --max_train_steps=50 \
        --max_adv_train_steps=6 \
        --pgd_alpha=5e-3 \
        --pgd_eps=0.12549019607843137
        
done 
