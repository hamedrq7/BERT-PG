import json

with open('dev.json', "r") as f:
    obj = json.load(f)
    print(type(obj))