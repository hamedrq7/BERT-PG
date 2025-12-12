BASE_DIR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/phase2paramfinding"

TASK_NAME=sst2
num_gpus=1
batch_size=32

# Safety checks
if [[ -z "$BASE_DIR" ]]; then
  echo "Usage: $0 /path/to/base_dir"
  exit 1
fi

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Error: '$BASE_DIR' is not a directory"
  exit 1
fi

# Loop over direct subdirectories only
for dir in "$BASE_DIR"/*/; do
  [[ -d "$dir" ]] || continue

  # Build the sodef_model path
  SODEF_MODEL_PATH="${dir%/}/phase3/phase3_best_acc_ckpt.pth"

  # Optional: skip if checkpoint does not exist
  if [[ ! -f "$SODEF_MODEL_PATH" ]]; then
    echo "Skipping (checkpoint not found): $SODEF_MODEL_PATH"
    continue
  fi

  echo "Running eval with sodef_model:"
  echo "  $SODEF_MODEL_PATH"

  torchrun --nproc_per_node=${num_gpus} \
    eval.py \
    --task_name "$TASK_NAME" \
    --sodef_model "$SODEF_MODEL_PATH" \
    --validation_file "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/dev.json" \
    --model_name_or_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2" \
    --max_length 128 \
    --pad_to_max_length \
    --per_device_eval_batch_size ${batch_size} \
    --seed 42 \
    --with_tracking \
    --sub_output_dir sodef_model-adv_glue/ \
    --eval_adv_glue

  torchrun --nproc_per_node=${num_gpus} \
    eval.py \
    --task_name "$TASK_NAME" \
    --sodef_model "$SODEF_MODEL_PATH" \
    --validation_file "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/dev.json" \
    --model_name_or_path "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2" \
    --max_length 128 \
    --pad_to_max_length \
    --per_device_eval_batch_size ${batch_size} \
    --seed 42 \
    --with_tracking \
    --sub_output_dir sodef_model-clean_glue \
    --eval_clean_glue
done