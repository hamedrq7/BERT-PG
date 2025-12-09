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

dims = weights['classifier.weight'].shape
print(dims)
print(type(weights['classifier.weight']))


import torch.nn as nn 
class BertCLF(nn.Module): 
    def __init__(self, dim_in, num_classes): 
        super(BertCLF, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(dim_in, num_classes)

    def forward(self, x): 
        return self.classifier(self.dropout(x))

dummy_model = BertCLF(dims[1], dims[0])
missing_keys, unexpected_keys = dummy_model.load_state_dict(weights)
print('missing_keys', missing_keys, 'len unexpected_keys', len(unexpected_keys))

torch.save(dummy_model.state_dict(), "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/bert_clf.pth")


"""
later do this: 
state_dict = new_model.state_dict()

state_dict["classifier.weight"] = weights["classifier.weight"]
state_dict["classifier.bias"] = weights["classifier.bias"]

new_model.load_state_dict(state_dict)
"""