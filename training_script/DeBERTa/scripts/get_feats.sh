TASK_NAME=sst2
EXP_INDEX=feats
num_epochs=5
warmup=40
lr=0e-1
num_gpus=1
batch_size=8
EXP_DIR="X"

torchrun --nproc_per_node=${num_gpus} \
  get_features_glue.py \
  --model_name_or_path $EXP_DIR \
  --config_name $EXP_DIR \
  --tokenizer_name $EXP_DIR \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --num_train_epochs ${num_epochs} \
  --warmup_steps ${warmup} \
  --learning_rate ${lr} \
  --per_device_train_batch_size ${batch_size} \
  --output_dir $EXP_DIR/$EXP_INDEX \
  --overwrite_output_dir \
  --logging_steps 10 \
  --logging_dir $EXP_DIR/$EXP_INDEX \
  --save_total_limit 1 
