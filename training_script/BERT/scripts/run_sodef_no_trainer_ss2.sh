TASK_NAME=sst2
num_gpus=1
batch_size=32

torchrun --nproc_per_node=${num_gpus} \
  wSodef_run_glue_no_trainer.py \
  --task_name $TASK_NAME \
  --validation_file '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/dev.json' \
  --train_file '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/dev.json' \
  --model_name_or_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2' \
  --sodef_model '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/SODEF_stuff/SODEF-PG/testingBertSodef/duos/' \
  --max_length 128 \
  --pad_to_max_length \
  --per_device_eval_batch_size ${batch_size} \
  --seed 42 \
  --with_tracking \
  --output_dir ./models/no_trainer/$TASK_NAME/adv_glue_sodef/ \
  --eval_adv_glue