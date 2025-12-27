import torch 
import torch.nn as nn 
import numpy as nn 
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np 

class BERT_feature_dataset(Dataset): 
    def __init__(self, x_np, y_np, ):
        self.x = torch.from_numpy(x_np).float()  
        self.y = torch.from_numpy(y_np).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx,:]
        y = self.y[idx]
        

        return x,y

def get_feature_dataloader(args, batch_size): 

    if args.feature_set_dir is None: 
        train_feature_set = get_single_sst2_feature_dataset(args.train_feature_set_dir) # , K = 100
        test_feature_set = get_single_sst2_feature_dataset(args.test_feature_set_dir) # , K = 100
    else: 
        train_feature_set, test_feature_set = get_sst2_feature_dataset(args.feature_set_dir)

    train_feature_loader = DataLoader(
        train_feature_set,
        batch_size=batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    test_feature_loader = DataLoader(
        test_feature_set,
        batch_size=batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    return train_feature_loader, test_feature_loader

def get_adv_glue_feature_dataset(path: str, TEMP=False): 
    loaded_np = np.load(path)
    if TEMP: 
        val_ds = BERT_feature_dataset(loaded_np['feats'], loaded_np['labels'])
    else: 
        val_ds = BERT_feature_dataset(loaded_np['val_feats'], loaded_np['val_labels'])

    print('Adv GLUE DS: ', val_ds.x.shape, val_ds.y.shape)
    
    return val_ds

def get_single_sst2_feature_dataset(path: str, K = None): 
    loaded_np = np.load(path)
    if K is None:
        ds = BERT_feature_dataset(loaded_np['feats'], loaded_np['labels'])
    else: 
        ds = BERT_feature_dataset(loaded_np['feats'][:K], loaded_np['labels'][:K])
    print('DS: ', ds.x.shape, ds.y.shape)
    
    return ds

def get_sst2_feature_dataset(path: str): 
    loaded_np = np.load(path)
    # Keys: train_feats, train_labels, val_feats, val_labels
    tr_ds = BERT_feature_dataset(loaded_np['train_feats'], loaded_np['train_labels'])
    val_ds = BERT_feature_dataset(loaded_np['val_feats'], loaded_np['val_labels'])

    print('Train DS: ', tr_ds.x.shape, tr_ds.y.shape)
    print('Val DS: ', val_ds.x.shape, val_ds.y.shape)
    
    return tr_ds, val_ds



def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

if __name__=="__main__":
    path_to_test = '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/mnli/feats/train_features.npz'
    get_single_sst2_feature_dataset(path=path_to_test)
    path_to_test = '/mnt/data/hossein/Hossein_workspace/nips_cetra/hamed/BERT-PG/training_script/DeBERTa/models/DeBERTs/large/mnli/feats/test_features.npz'
    get_single_sst2_feature_dataset(path=path_to_test)
    