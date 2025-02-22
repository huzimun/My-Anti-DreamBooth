export EXPERIMENT_NAME="E-ASPL"
export CLASS_DIR="data/class-person"
export DATASET_DIR="/home/humw/Datasets/VGGFace2"
export device="cuda:3"

# ------------------------- Train E-ASPL on set B -------------------------
# pretrained sd models
sd14_path="/home/humw/Pretrains/stable-diffusion-v1-4"
sd15_path="/home/humw/Pretrains/stable-diffusion-v1-5"
sd21_path="/home/humw/Pretrains/stable-diffusion-2-1-base"
ref_model_path="${sd14_path},${sd15_path},${sd21_path}"

for person_id in `ls $DATASET_DIR`; do   
# for person_id in "n000050"; do   
  export CLEAN_TRAIN_DIR=${DATASET_DIR}"/"${person_id}"/set_A" 
  export CLEAN_ADV_DIR=${DATASET_DIR}"/"${person_id}"/set_B"
  export ADV_OUTPUT_DIR="outputs/adversarial_images/"$EXPERIMENT_NAME"/"${person_id}
  echo ${CLEAN_TRAIN_DIR}
  echo ${CLEAN_ADV_DIR}
  echo ${ADV_OUTPUT_DIR}
  mkdir -p $ADV_OUTPUT_DIR

  # accelerate launch attacks/ensemble_aspl.py \
  python3 attacks/ensemble_aspl.py \
    --device $device \
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