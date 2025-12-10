import json
import datasets 

validation_file = 'dev.json'
data_files = {}
if validation_file is not None:
    data_files["validation"] = validation_file
extension = (validation_file).split(".")[-1]
raw_datasets = datasets.load_dataset(extension, data_files=data_files, field='sst2')
print(extension, data_files)
print(type(raw_datasets))
print(raw_datasets.keys())
print(type(raw_datasets['validation']))
print(raw_datasets['validation'].features)
print(raw_datasets["validation"][0]["sst2"][:5])
