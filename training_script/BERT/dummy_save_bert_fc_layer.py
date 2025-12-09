import torch
from safetensors.torch import load_file

weights = load_file("/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/model.safetensors")
print(weights.keys())