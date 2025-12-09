TASK_NAME=sst2
num_epochs=4
warmup=40
lr=2e-5
num_gpus=2
batch_size=32

torchrun --nproc_per_node=${num_gpus} \
  run_glue_no_trainer.py \
  --model_name_or_path bert-large-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --pad_to_max_length \
  --num_train_epochs ${num_epochs} \
  --num_warmup_steps ${warmup} \
  --learning_rate ${lr} \
  --per_device_train_batch_size ${batch_size} \
  --output_dir ./models/no_trainer/$TASK_NAME/ \
  --checkpointing_steps 'epoch' \
  --seed 42 \
  --with_tracking

TASK_NAME=qqp
num_epochs=4
warmup=64
lr=2e-5
num_gpus=1
batch_size=32

torchrun --nproc_per_node=${num_gpus} \
  run_glue_no_trainer.py \
  --model_name_or_path bert-large-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --pad_to_max_length \
  --num_train_epochs ${num_epochs} \
  --num_warmup_steps ${warmup} \
  --learning_rate ${lr} \
  --per_device_train_batch_size ${batch_size} \
  --output_dir ./models/no_trainer/$TASK_NAME/ \
  --checkpointing_steps 'epoch' \
  --seed 42 \
  --with_tracking