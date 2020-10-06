import os
import torch
import numpy as np

def save_model(model, path, name):
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, name))

def convert_multimodal_mnist(dataset, except_class):
    '''
    mnist 10개 클래스 중 특정 클래스 데이터를 제외
    '''
    total_class = [i for i in range(10) if i != except_class]

    normal_cls_idx = np.where(np.isin(dataset.targets, total_class))
    dataset.data = dataset.data[normal_cls_idx[0]]
    dataset.targets = dataset.targets[normal_cls_idx[0]]
    return dataset