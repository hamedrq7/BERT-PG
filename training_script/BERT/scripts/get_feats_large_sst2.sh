TASK_NAME=sst2
EXP_INDEX=test
num_epochs=5
warmup=40
lr=0e-1
num_gpus=1
batch_size=32

python -m torch.distributed.launch --nproc_per_node=${num_gpus} \
  run_glue.py \
  --model_name_or_path '/content/adversarial-glue/training_script/BERT/models/sst2/train0' \
  --config_name '/content/adversarial-glue/training_script/BERT/models/sst2/train0' \
  --tokenizer_name '/content/adversarial-glue/training_script/BERT/models/sst2/train0' \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --num_train_epochs ${num_epochs} \
  --warmup_steps ${warmup} \
  --learning_rate ${lr} \
  --per_device_train_batch_size ${batch_size} \
  --output_dir ./models/$TASK_NAME/$EXP_INDEX \
  --overwrite_output_dir \
  --logging_steps 10 \
  --logging_dir ./models/$TASK_NAME/$EXP_INDEX \
  --save_total_limit 1 \
  --max_train_samples 100 \
  --max_val_samples 100