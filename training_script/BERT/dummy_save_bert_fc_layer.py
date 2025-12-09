import torch
from safetensors.torch import load_file
from transformers.models.bert import BertForSequenceClassification
"""
For bert-base-cased:
.... 'bert.encoder.layer.9.intermediate.dense.bias', 'bert.encoder.layer.9.intermediate.dense.weight', 'bert.encoder.layer.9.output.LayerNorm.bias', 'bert.encoder.layer.9.output.LayerNorm.weight', 'bert.encoder.layer.9.output.dense.bias', 'bert.encoder.layer.9.output.dense.weight', 'bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'classifier.bias', 'classifier.weight']
"""

# ******* the nn.Dropout does not have parameters ********** # Dropout(p=0.1, inplace=False)

weights = load_file("/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/model.safetensors")
print(weights.keys())
print(type(weights['classifier.weight']))
print(type(weights['dropout']))


# import torch.nn as nn 
# class BertCLF(nn.module): 
#     def __init__(self, ): 
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#     pass


"""
later do this: 
state_dict = new_model.state_dict()

state_dict["classifier.weight"] = weights["classifier.weight"]
state_dict["classifier.bias"] = weights["classifier.bias"]

new_model.load_state_dict(state_dict)
"""