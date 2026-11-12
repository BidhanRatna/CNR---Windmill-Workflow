import importlib.util
import json

modules = [
    "numpy",
    "pandas",
    "torch",
    "torchvision",
    "torch_dct",
    "foolbox",
    "tqdm"
]

results = {}

for module in modules:
    results[module] = importlib.util.find_spec(module) is not None

print(json.dumps(results, indent=2))
