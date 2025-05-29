from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os


def load_data(data_dir, batch_size):
    """
    载入数据
    参数：
        data_dir: 源数据目录路径
        batch_size: batch大小
    """
    # 1. 数据增强与归一化
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 创建数据集
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), val_transform)

    # 3. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"类别数量: {len(train_dataset.classes)}")
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(val_dataset)}")

    return train_loader, val_loader
