import torch
from safetensors.torch import load_file
from transformers.models.deberta import DebertaForSequenceClassification
"""
For bert-base-cased:
.... 'bert.encoder.layer.9.intermediate.dense.bias', 'bert.encoder.layer.9.intermediate.dense.weight', 'bert.encoder.layer.9.output.LayerNorm.bias', 'bert.encoder.layer.9.output.LayerNorm.weight', 'bert.encoder.layer.9.output.dense.bias', 'bert.encoder.layer.9.output.dense.weight', 'bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'classifier.bias', 'classifier.weight']
"""

# ******* the nn.Dropout does not have parameters ********** # Dropout(p=0.1, inplace=False)

weights = load_file("/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/mnli/model.safetensors")
print(weights.keys())

dims = weights['classifier.weight'].shape
print(dims)
print(type(weights['classifier.weight']))


import torch.nn as nn 
class DeBertCLF(nn.Module): 
    def __init__(self, dim_in, num_classes): 
        super(DeBertCLF, self).__init__()
        self.classifier = nn.Linear(dim_in, num_classes)

    def forward(self, x): 
        return self.classifier((x))

dummy_model = DeBertCLF(dims[1], dims[0])
missing_keys, unexpected_keys = dummy_model.load_state_dict(weights, strict=False)
print('missing_keys', missing_keys, 'len unexpected_keys', len(unexpected_keys))

from SODEF.data_utils import get_single_sst2_feature_dataset
root = '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/mnli/feats'


def accu(model, loader, device, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):            
            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs
            outputs = model(x)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    avg_loss = test_loss/(batch_idx+1)
    acc = correct/total
    return {
        'acc': acc, 
        'loss': avg_loss,
    }

adv_m = get_single_sst2_feature_dataset(f'{root}/adv_glue-m_features.npz')
adv_mm = get_single_sst2_feature_dataset(f'{root}/adv_glue-mm_features.npz')
tr = get_single_sst2_feature_dataset(f'{root}/train_features.npz')
te_mm = get_single_sst2_feature_dataset(f'{root}/test-mm_features.npz')
te_m = get_single_sst2_feature_dataset(f'{root}/test-m_features.npz')
device = torch.device(f'cuda:{0}')

from torch.utils.data import DataLoader
def _test(ds, name, model, device): 
    print('Testing ', name)
    dl = DataLoader(ds, batch_size=128)

    te_res = accu(model, dl, device, nn.CrossEntropyLoss())
    print('Acc, Loss', te_res['acc'], te_res['loss'])

_test(tr, 'train', dummy_model, device)
_test(te_m, 'test-m', dummy_model, device)
_test(te_mm, 'test-mm', dummy_model, device)
_test(adv_m, 'adv-m', dummy_model, device)
_test(adv_mm, 'adv-mm', dummy_model, device)

# torch.save(dummy_model.state_dict(), "/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/BERT/models/no_trainer/sst2/bert_clf.pth")


"""
later do this: 
state_dict = new_model.state_dict()

state_dict["classifier.weight"] = weights["classifier.weight"]
state_dict["classifier.bias"] = weights["classifier.bias"]

new_model.load_state_dict(state_dict)
"""