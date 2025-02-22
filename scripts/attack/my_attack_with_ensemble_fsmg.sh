export EXPERIMENT_NAME="E-FSMG"
export CLASS_DIR="data/class-person"
export DATASET_DIR="/home/humw/Datasets/VGGFace2"


# ------------------------- Train DreamBooth models on set A -------------------------
# stable diffusion
sd14_path="./stable-diffusion/stable-diffusion-v1-4"
sd15_path="./stable-diffusion/stable-diffusion-v1-5"
sd21_path="./stable-diffusion/stable-diffusion-2-1-base"
sd_paths=($sd14_path $sd15_path $sd21_path)

# versions=(V14 V15 V21)

for person_id in `ls $DATASET_DIR`; do   
# for person_id in "n000050"; do 
  for ((i=0;i<3;i++));
  do
      export CLEAN_TRAIN_DIR=${DATASET_DIR}"/"${person_id}"/set_A" 
      # ref models
      ref_sd14_root="./dreambooth-clean-outputs/V14_VGG512_set_A/"${person_id}"_REFERENCE/"
      ref_sd15_root="./dreambooth-clean-outputs/V15_VGG512_set_A/"${person_id}"_REFERENCE/"
      ref_sd21_root="./dreambooth-clean-outputs/V21_VGG512_set_A/"${person_id}"_REFERENCE/"
      ref_paths=($ref_sd14_root $ref_sd15_root $ref_sd21_root)
      echo ${ref_paths[$i]}
      echo ${sd_paths[$i]}
      # echo ${versions[$i]}

      accelerate launch train_dreambooth.py \
          --pretrained_model_name_or_path=${sd_paths[$i]}  \
          --enable_xformers_memory_efficient_attention \
          --train_text_encoder \
          --instance_data_dir=$CLEAN_TRAIN_DIR \
          --class_data_dir=$CLASS_DIR \
          --output_dir=${ref_paths[$i]} \
          --with_prior_preservation \
          --prior_loss_weight=1.0 \
          --instance_prompt="a photo of sks person" \
          --class_prompt="a photo of person" \
          --inference_prompt="a photo of sks person;a dslr portrait of sks person" \
          --resolution=512 \
          --train_batch_size=1 \
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
          --sample_batch_size=16
  done


  # ------------------------- Train E-FSMG on set B -------------------------
  export CLEAN_ADV_DIR=${DATASET_DIR}"/"${person_id}"/set_B"
  export ADV_OUTPUT_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME"/"${person_id}

  mkdir -p $ADV_OUTPUT_DIR
  # cp -r $CLEAN_ADV_DIR $OUTPUT_DIR/image_before_addding_noise

  # pretrained sd models on clean set A
  ref_sd14_path="${ref_sd14_root}/checkpoint-1000"
  ref_sd15_path="${ref_sd15_root}/checkpoint-1000"
  ref_sd21_path="${ref_sd21_root}/checkpoint-1000"
  ref_model_paths="${ref_sd14_path},${ref_sd15_path},${ref_sd21_path}"

  accelerate launch attacks/ensemble_fsmg.py \
    --pretrained_model_name_or_path=${ref_model_paths} \
    --enable_xformers_memory_efficient_attention \
    --train_text_encoder \
    --instance_data_dir=$CLEAN_ADV_DIR \
    --output_dir=$ADV_OUTPUT_DIR \
    --instance_prompt="a photo of sks person" \
    --resolution=512 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=60 \
    --max_adv_train_steps=12 \
    --checkpointing_steps=20 \
    --center_crop \
    --pgd_alpha=5e-3 \
    --pgd_eps=0.12549019607843137

  # # 定义目标目录
  # TARGET_DIR="dreambooth-clean-outputs"

  # # 检查目标目录是否存在
  # if [ ! -d "$TARGET_DIR" ]; then
  #     echo "目标目录 $TARGET_DIR 不存在."
  #     exit 1
  # fi

  # # 打印将要删除的路径，并询问用户是否确定
  # echo "即将删除目录 $TARGET_DIR 下的所有文件和文件夹："
  # # read -p "您确定要继续吗？(y/n): " CONFIRM

  # # if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
  # #     echo "已取消删除操作."
  # #     exit 0
  # # fi

  # # 迭代删除所有子文件和文件夹
  # rm -rf "$TARGET_DIR"/*
  # rmdir "$TARGET_DIR"

  # # 提示删除完成
  # echo "目录 $TARGET_DIR 下的所有文件和文件夹已成功删除."
done
