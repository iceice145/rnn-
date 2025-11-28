import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"模型已保存到: {path}")

def load_model(model, path, device=None):
    if device is None:
        device = get_device()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"模型已从 {path} 加载")
    return model

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, title="训练曲线"):
    """绘制训练/验证损失和准确率曲线"""
    plt.figure(figsize=(12, 5))
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='训练损失')
    plt.plot(val_losses, 'r-', label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title(f'{title} - 损失对比')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='训练准确率')
    plt.plot(val_accs, 'r-', label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.title(f'{title} - 准确率对比')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')
    plt.show()


# def plot_confusion_matrix(y_true, y_pred, classes, title="混淆矩阵"):
#     """混淆矩阵"""
#     cm = confusion_matrix(y_true, y_pred)
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(
#         cm_normalized,
#         annot=True,
#         fmt='.2f',
#         cmap='Blues',
#         xticklabels=classes,
#         yticklabels=classes
#     )
#     plt.xlabel('预测类别')
#     plt.ylabel('真实类别')
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')
#     plt.show()

# 简单计时器
class Timer:
    def __init__(self):
        self.start_time = None
    def start(self):
        self.start_time = time.time()
        return self
    def stop(self):
        return time.time() - self.start_time if self.start_time else 0

# 时间格式化
# def format_time(seconds):
#     if seconds < 60:
#         return f"{seconds:.2f}秒"
#     elif seconds < 3600:
#         minutes = seconds // 60
#         seconds = seconds % 60
#         return f"{int(minutes)}分{seconds:.2f}秒"
#     else:
#         hours = seconds // 3600
#         minutes = (seconds % 3600) // 60
#         seconds = seconds % 60
#         return f"{int(hours)}时{int(minutes)}分{seconds:.2f}秒"


def print_model_info(model, model_name):
    print(f"\n{model_name} 信息:")
    print(f"  参数数量: {count_parameters(model):,}")
    print(f"  结构: {model.__class__.__name__}")


def evaluate_model(model, dataloader, device, classes):
    """完整评估模型（准确率+混淆矩阵+分类报告）"""
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"\n准确率: {accuracy:.2f}%")
    print("\n分类报告:")
    print(classification_report(
        y_true, y_pred, target_names=classes, zero_division=0
    ))
    # plot_confusion_matrix(y_true, y_pred, classes)

    model.train()
    return accuracy