TASK_NAME=sst2
EXP_INDEX=feats
num_epochs=7
warmup=10000
lr=0e-1
num_gpus=1
batch_size=1
EXP_DIR="/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/sst2"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=${num_gpus} --master_port=29501 \
  get_features_glue.py \
  --model_name_or_path $EXP_DIR \
  --config_name $EXP_DIR \
  --tokenizer_name $EXP_DIR \
  --task_name $TASK_NAME \
  --do_eval \
  --max_seq_length 128 \
  --num_train_epochs ${num_epochs} \
  --warmup_steps ${warmup} \
  --learning_rate ${lr} \
  --per_device_train_batch_size ${batch_size} \
  --per_device_eval_batch_size 1 \
  --output_dir $EXP_DIR/$EXP_INDEX \
  --overwrite_output_dir \
  --logging_steps 10 \
  --logging_dir $EXP_DIR/$EXP_INDEX \
  --save_total_limit 1 \
  --eval_accumulation_steps 1 \
  --max_val_samples 5 \

#   --max_train_samples 100 \