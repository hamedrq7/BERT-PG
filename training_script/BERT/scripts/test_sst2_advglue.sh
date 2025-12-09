TASK_NAME=sst2
num_gpus=1
batch_size=32

torchrun --nproc_per_node=${num_gpus} \
  run_glue_no_trainer.py \
  --validation_file 'dev.json' \
  --model_name_or_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2' \
  --task_name $TASK_NAME \
  --max_length 128 \
  --pad_to_max_length \
  --per_device_eval_batch_size ${batch_size} \
  --seed 42 \
  --with_tracking \
  --output_dir ./models/no_trainer/$TASK_NAME/adv_glue/ \
  --only_eval