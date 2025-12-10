TASK_NAME=sst2
num_gpus=1
batch_size=32

# Adv base model
torchrun --nproc_per_node=${num_gpus} \
  eval.py \
  --task_name $TASK_NAME \
  --validation_file '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/dev.json' \
  --model_name_or_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2' \
  --max_length 128 \
  --pad_to_max_length \
  --per_device_eval_batch_size ${batch_size} \
  --seed 42 \
  --with_tracking \
  --sub_output_dir ./base_model/adv_glue/ \
  --eval_adv_glue

# # Adv Sodef Model
# torchrun --nproc_per_node=${num_gpus} \
#   eval.py \
#   --task_name $TASK_NAME \
#   --sodef_model '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/SODEF_stuff/SODEF-PG/testingBertSodef/duos/phase3_best_acc_ckpt.pth' \
#   --validation_file '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/dev.json' \
#   --model_name_or_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2' \
#   --max_length 128 \
#   --pad_to_max_length \
#   --per_device_eval_batch_size ${batch_size} \
#   --seed 42 \
#   --with_tracking \
#   --sub_output_dir sodef_model/adv_glue/ \
#   --eval_adv_glue

# # Clean base model
# torchrun --nproc_per_node=${num_gpus} \
#   eval.py \
#   --task_name $TASK_NAME \
#   --model_name_or_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2' \
#   --max_length 128 \
#   --pad_to_max_length \
#   --per_device_eval_batch_size ${batch_size} \
#   --seed 42 \
#   --with_tracking \
#   --sub_output_dir base_model/clean_glue/ \
#   --eval_clean_glue


# # SODEF base model
# torchrun --nproc_per_node=${num_gpus} \
#   eval.py \
#   --task_name $TASK_NAME \
#   --sodef_model '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/SODEF_stuff/SODEF-PG/testingBertSodef/duos/phase3_best_acc_ckpt.pth' \
#   --model_name_or_path '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2' \
#   --max_length 128 \
#   --pad_to_max_length \
#   --per_device_eval_batch_size ${batch_size} \
#   --seed 42 \
#   --with_tracking \
#   --sub_output_dir base_model/clean_glue/ \
#   --eval_clean_glue