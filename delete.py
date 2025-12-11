# import json
# import datasets 

# validation_file = 'dev.json'
# data_files = {}
# if validation_file is not None:
#     data_files["validation"] = validation_file
# extension = (validation_file).split(".")[-1]
# raw_datasets = datasets.load_dataset(extension, data_files=data_files, field='sst2')
# print(extension, data_files)
# print(type(raw_datasets))
# print(raw_datasets.keys())
# print(type(raw_datasets['validation']))
# print(raw_datasets['validation'].features)
# print(raw_datasets["validation"][0]["sst2"][:5])

import torch

# Optimizer with initial LR = 1.0 (easy to see changes)
optimizer = torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=1.0)

total_iters = 20
decay_iter = 10   # decay at step 10 (50% of training)

def lr_lambda(step):
    if step >= decay_iter:
        return 0.1
    else:
        return 1.0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print("Step | LR")
print("--------------")

for step in range(total_iters):
    # optimizer step (dummy)
    optimizer.step()
    
    # scheduler step
    scheduler.step()

    # print LR
    lr = optimizer.param_groups[0]['lr']
    print(f"{step:4d} | {lr:.6f}")