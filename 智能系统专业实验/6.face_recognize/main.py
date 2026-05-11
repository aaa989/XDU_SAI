import os
import time
import torch
import torch_pruning as tp
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import utils
from dataset import read_split_data, MyDataSet
from model import MyModel

def main():
    # 初始化配置与环境
    config = utils.read_config("config.json")
    utils.set_seed(config['seed'])
    os.makedirs(config['save_path'], exist_ok=True)
    tb_writer = SummaryWriter(log_dir=f"logs/{time.strftime('%Y_%m_%d_%H_%M_%S')}")

    # 数据预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((config['img_size'], config['img_size'])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(config['img_size'], padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((config['img_size'], config['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 数据集加载
    train_images_path, train_images_label, _, _ = read_split_data(config['train_data_root'], config['val_rate'])
    test_images_path, test_images_label, _, _ = read_split_data(config['test_data_root'], 0.0)

    train_dataset = MyDataSet(train_images_path, train_images_label, transform=data_transform["train"])
    test_dataset = MyDataSet(test_images_path, test_images_label, transform=data_transform["test"])

    nw = min([os.cpu_count(), config['batch_size'] if config['batch_size'] > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset, config['batch_size'], shuffle=True, num_workers=nw, pin_memory=True, drop_last=True, collate_fn=train_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, config['batch_size'], shuffle=False, num_workers=nw, pin_memory=True, collate_fn=test_dataset.collate_fn)

    # 模型与优化器
    model = MyModel(num_classes=config['num_class'], pretrained=True).to(config['device'] if torch.cuda.is_available() else "cpu")

    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=config['weight_decay'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0)

    # 训练循环
    best_acc = 0.0
    for epoch in range(config['epochs']):
        train_loss, train_acc = utils.trainer(model, optimizer, train_loader, config, epoch)
        test_loss, test_acc = utils.evaluater(model, test_loader, config, epoch)

        tb_writer.add_scalar('train_loss', train_loss, epoch)
        tb_writer.add_scalar('train_acc', train_acc, epoch)
        tb_writer.add_scalar('test_loss', test_loss, epoch)
        tb_writer.add_scalar('test_acc', test_acc, epoch)
        tb_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'acc': test_acc}, os.path.join(config['save_path'], "best_model.pth"))

        scheduler.step()

    # 模型剪枝
    checkpoint = torch.load(os.path.join(config['save_path'], "best_model.pth"), map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])

    ignored_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear) and m.out_features == config['num_class']]
    example_inputs = torch.randn(1, 3, config['img_size'], config['img_size']).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    pruner = tp.pruner.MetaPruner(model, example_inputs, importance=utils.MySlimmingImportance(), iterative_steps=config['epochs'], ch_sparsity=0.3, ignored_layers=ignored_layers)
    for _ in range(config['epochs']):
        pruner.step()

    torch.save(model, os.path.join(config['save_path'], "pruned_model_full.pth"))

if __name__ == '__main__':
    main()