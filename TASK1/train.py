import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data_load import load_data
from model import MyModel
import os


def train(config, best_acc):
    """
    模型训练与评估
    参数：
        config: 参数配置
    """
    # 1. 数据载入
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    dir_name = f'{"pretrained" if config["pretrained"] else "scratch"} + batch {config["batch_size"]} +  lr {config["lr"]} + fc {config["lr_fc"]}'
    writer = SummaryWriter(os.path.join('tensor_board', dir_name))

    train_loader, val_loader = load_data(config['data_dir'], config['batch_size'])

    # 3. 模型创建
    tmp = MyModel(pretrained=config['pretrained'])
    model = tmp.model.to(device)

    # 4. 优化器&学习率&优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    if config['pretrained']:
        params = tmp.parameters_tuning(config['lr'], config['lr_fc'])
    else:
        params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=config['lr'])  # Adam优化器

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])  # 余弦退火策略

    # 5.训练与测试
    for epoch in range(config['epochs']):
        train_loss, train_correct, train_total = 0, 0, 0
        val_loss, val_correct, val_total = 0, 0, 0

        # 5.1 模型训练
        model.train()
        for _, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()  # 更新学习率

        # 5.2 模型评估
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # 5.3 指标计算与保存
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch: {epoch+1}/{config["epochs"]} | '
              f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        # 5.5 模型保存
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join('checkpoint', f'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Best accuracy: {best_acc:.2f}%')
            print(f'Model saved to {checkpoint_path}')

    # 6. 关闭Tensorboard
    writer.close()
    return best_acc

