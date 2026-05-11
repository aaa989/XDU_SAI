import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# 数据集路径读取与划分（无验证集）
def read_split_data(root: str, val_rate: float = 0.0):
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []

    classes = sorted([entry.name for entry in os.scandir(root) if entry.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        cls_path = os.path.join(root, cls)
        images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if
                  img.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(class_to_idx[cls])

    return train_images_path, train_images_label, val_images_path, val_images_label


# 自定义数据集类
class MyDataSet(Dataset):
    def __init__(self, images_path, images_class, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.images_class[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels