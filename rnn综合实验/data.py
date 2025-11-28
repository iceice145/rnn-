import torch
import unicodedata
import string
import glob
import random
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 字符集扩展（增加常见特殊字符）
ALL_LETTERS = string.ascii_letters + " .,;'-"
N_LETTERS = len(ALL_LETTERS) + 1  # +1 用于padding

def unicodeToAscii(s):
    """Unicode转ASCII，过滤无效字符"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

def readLines(filename):
    """读取文件并处理"""
    with open(filename, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# 加载所有数据
category_lines = {}
all_categories = []
for filename in glob.glob('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

all_names = [name for cat in category_lines.values() for name in cat]
name_lengths = [len(name) for name in all_names]
MAX_LENGTH = max(name_lengths)
print(f"姓名长度统计：平均{sum(name_lengths)/len(name_lengths):.1f}，最长{MAX_LENGTH}，最短{min(name_lengths)}")

# 类别平衡统计
print("\n类别分布：")
for cat in all_categories:
    print(f"{cat}: {len(category_lines[cat])}个样本")

def nameToTensor(name, max_len=MAX_LENGTH):
    """将姓名转为张量（padding到固定长度）"""
    tensor = torch.zeros(max_len, N_LETTERS)  # (max_len, N_LETTERS)
    for i, letter in enumerate(name[:max_len]):  # 截断过长姓名
        if letter in ALL_LETTERS:
            tensor[i][ALL_LETTERS.find(letter)] = 1
    return tensor

class SurnameDataset(Dataset):
    """姓氏数据集类（支持划分训练/验证/测试集）"""
    def __init__(self, data, labels):
        self.data = data  # 姓名张量列表
        self.labels = labels  # 类别索引列表

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_datasets(test_size=0.2, val_size=0.1, batch_size=32):
    # 整理所有数据
    all_data = []
    all_labels = []
    for cat_idx, cat in enumerate(all_categories):
        for name in category_lines[cat]:
            tensor = nameToTensor(name)
            all_data.append(tensor)
            all_labels.append(cat_idx)

    # 划分训练+验证集 与 测试集
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        all_data, all_labels, test_size=test_size, random_state=42, stratify=all_labels  # stratify保证类别分布一致
    )

    # 划分训练集与验证集
    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=val_size/(1-test_size), random_state=42, stratify=train_val_labels
    )

    # 创建Dataset和DataLoader
    train_dataset = SurnameDataset(train_data, train_labels)
    val_dataset = SurnameDataset(val_data, val_labels)
    test_dataset = SurnameDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n数据集划分：")
    print(f"训练集：{len(train_dataset)}个样本")
    print(f"验证集：{len(val_dataset)}个样本")
    print(f"测试集：{len(test_dataset)}个样本")
    return train_loader, val_loader, test_loader

__all__ = [
    'ALL_LETTERS', 'N_LETTERS', 'n_categories', 'all_categories',
    'MAX_LENGTH', 'unicodeToAscii', 'nameToTensor', 'create_datasets'
]