import os
import shutil
from sklearn.model_selection import train_test_split


def split_data(data_root='101_ObjectCategories', train_ratio=0.8):
    """
    将图像数据集分割为训练集和测试集
    参数:
        data_dir: 源数据目录路径
        train_ratio: 训练集比例
    """
    # 1. 训练集与测试集目录
    train_dir, test_dir = 'data/train', 'data/test'
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 2. 图片种类
    categories = [d for d in os.listdir(data_root)]

    # 3. 训练集与测试集划分
    for category in categories:
        # 3.1 图片来源
        src_dir = os.path.join(data_root, category)
        # 3.2 图片分类
        images = [f for f in os.listdir(src_dir)]
        train_imgs, test_imgs = train_test_split(images, train_size=train_ratio, random_state=2002)
        # 3.3 图片复制
        for phase, imgs in [(train_dir, train_imgs), (test_dir, test_imgs)]:
            dest_dir = os.path.join(phase, category)
            os.makedirs(dest_dir, exist_ok=True)
            for img in imgs:
                shutil.copy2(os.path.join(src_dir, img), os.path.join(dest_dir, img))


if __name__ == '__main__':
    split_data()
    print("数据分割完成!!")
