import os
import torch

def save_model(model, path, name):
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, name))
