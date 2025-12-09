import torch
from safetensors.torch import load_file

weights = load_file("/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/model.safetensors")
print(weights.keys())
print(type(weights['classifier']))
print(type(weights['dropout']))

"""
later do this: 
state_dict = new_model.state_dict()

state_dict["classifier.weight"] = weights["classifier.weight"]
state_dict["classifier.bias"] = weights["classifier.bias"]

new_model.load_state_dict(state_dict)
"""