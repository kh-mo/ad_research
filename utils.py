import os
import copy
import torch
import numpy as np

def save_model(model, path, name):
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, name))

def convert_multimodal_mnist(dset, except_class):
    '''
        mnist 10개 클래스 중 특정 클래스 데이터를 제외
        단, 테스트때는 모든 class 사용
    '''
    dataset = copy.deepcopy(dset)
    if except_class is None:
        total_class = [i for i in range(10)]
    else:
        total_class = [i for i in range(10) if i != except_class]

    normal_cls_idx = np.where(np.isin(dataset.targets, total_class))
    dataset.data = dataset.data[normal_cls_idx[0]]
    dataset.targets = dataset.targets[normal_cls_idx[0]]
    return dataset

def show_online_image(dset, except_class, amount):
    '''
        중간중간 생성되는 이미지를 보여주기 위한 데이터셋
    '''
    dataset = copy.deepcopy(dset)
    total_class = [i for i in range(10) if i != except_class]

    online_img_idx = []
    for i in total_class:
        online_img_idx += np.where(np.isin(dataset.targets, i))[0][:amount].tolist()

    dataset.data = dataset.data[online_img_idx]
    dataset.targets = dataset.targets[online_img_idx]
    return dataset

def print_score(path):
    '''
    저장된 모델 이름보고 score 찍기
    '''
    from itertools import groupby

    # files = os.listdir(os.path.join(os.getcwd(), "save_model/ae"))
    files = os.listdir(os.path.join(os.getcwd(), path))
    group = groupby([i.split("_") for i in files], lambda x: x[0])

    class_score = {"all": []}
    for key, items in group:
        class_score[key] = []
        for item in items:
            auc = float(item[-1].replace("auroc", ""))
            class_score[key].append(auc)
            class_score["all"].append(auc)
        print("{} : {}({})".format(key,
                                   round(np.average(class_score[key]), 3),
                                   round(np.std(class_score[key]),3)
                                   ))
    print("All : {}({})".format(round(np.average(class_score["all"]), 3),
                                round(np.std(class_score[key]), 3)
                                ))



