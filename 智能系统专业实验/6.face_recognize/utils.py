import json
import os
import random
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch_pruning as tp

# 读取模型配置文件
def read_config(config_path="config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config['model_config']

# 固定随机种子，保证实验可复现
def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 单轮训练
def trainer(model, optimizer, data_loader, config, epoch):
    device = config['device']
    model.train()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, desc=f"Train Epoch {epoch}")
    criterion = nn.CrossEntropyLoss()

    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        pred_classes = torch.max(outputs, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum()
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        accu_loss += loss.detach()
        sample_num += images.shape[0]
        data_loader.desc = f"[train epoch {epoch}] loss: {accu_loss.item()/(step+1):.3f}, acc: {accu_num.item()/sample_num:.3f}"

        if not torch.isfinite(loss):
            sys.exit(1)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

# 单轮评估
def evaluater(model, data_loader, config, epoch):
    device = config['device']
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout, desc=f"Test Epoch {epoch}")
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for step, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            pred_classes = torch.max(outputs, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum()
            accu_loss += loss.detach()
            sample_num += images.shape[0]
            
            data_loader.desc = f"[test epoch {epoch}] loss: {accu_loss.item()/(step+1):.3f}, acc: {accu_num.item()/sample_num:.3f}"

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

# 剪枝重要性评估
class MySlimmingImportance(tp.importance.Importance):
    def __call__(self, group, **kwargs):
        group_imp = []
        for dep, idxs in group:
            layer = dep.target.module
            if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and layer.affine:
                local_imp = torch.abs(layer.weight.data)
                group_imp.append(local_imp)
        if len(group_imp) == 0:
            return None
        group_imp = torch.stack(group_imp, dim=0).mean(dim=0)
        return group_imp

# BN层稀疏正则化
def add_sparse_regularization(model, reg=1e-4):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.affine:
            if m.weight.grad is not None:
                m.weight.grad.data.add_(reg * torch.sign(m.weight.data))