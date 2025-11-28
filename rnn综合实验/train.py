import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import create_datasets, N_LETTERS, n_categories, all_categories, nameToTensor, unicodeToAscii, MAX_LENGTH
from model import CharRNN, CharGRU, CharLSTM
from utils import *

# 设备配置
device = get_device()
print(f"使用设备: {device}")

# 超参数配置
INPUT_SIZE = N_LETTERS
HIDDEN_SIZE = 256
OUTPUT_SIZE = n_categories
NUM_LAYERS = 3
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5  # L2正则化


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # 反向传播+op优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 统计损失和准确率
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def val_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_model(model, model_name, train_loader, val_loader, test_loader):
    """完整训练流程"""
    print(f"\n=== 开始训练 {model_name} ===")
    print_model_info(model, model_name)

    model = model.to(device)
    criterion = nn.NLLLoss()

    # 优化器：Adam + L2正则化
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    best_val_loss = float('inf')
    timer = Timer().start()
    current_lr = LEARNING_RATE

    for epoch in range(EPOCHS):
        # 训练和验证
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)

        # 学习率调度 + 手动打印调整信息
        old_lr = current_lr
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            print(f"=== 学习率调整：{old_lr:.6f} → {current_lr:.6f} ===")

        # 保存训练信息
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            save_model(model, f"best_{model_name}.pth")
        # 打印训练过程的信息
        print(f"Epoch {epoch + 1:3d}/{EPOCHS} | "
              f"损失: {train_loss:.4f} | 准确率: {train_acc:.2f}% | ")
    # 训练完成
    training_time = timer.stop()
    print(f"\n{model_name} 训练完成! 用时: {format_time(training_time)}")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (验证损失: {best_val_loss:.4f})")
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, f"{model_name}训练曲线")
    # 加载最优模型并测试
    print(f"\n=== 测试 {model_name} 最优模型 ===")
    best_model = load_model(model, f"best_{model_name}.pth", device)
    test_acc = evaluate_model(best_model, test_loader, device, all_categories)
    print(f"测试准确率: {test_acc:.2f}%")
    return best_model, test_acc


def predict(model, name, all_categories, device):
    model.eval()
    name = unicodeToAscii(name)
    tensor = nameToTensor(name, MAX_LENGTH).unsqueeze(0).to(device)  # (1, max_len, input_size)
    with torch.no_grad():
        outputs = model(tensor)
        topv, topi = outputs.topk(3, 1, True)
    print(f"\n> {name}")
    for i in range(3):
        prob = torch.exp(topv[0][i]).item() * 100
        cat_idx = topi[0][i].item()
        print(f"  {all_categories[cat_idx]} ({prob:.1f}%)")


if __name__ == "__main__":
    # 创建数据集
    train_loader, val_loader, test_loader = create_datasets(batch_size=BATCH_SIZE)

    model = CharLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    # 训练模型
    trained_model, test_acc = train_model(
        model, "CharLSTM", train_loader, val_loader, test_loader
    )
    # 测试预测
    print("\n=== 预测测试 ===")
    test_names = ["Alice", "Zhang", "Wang", "Smith", "Yamamoto", "Garcia", "Muller", "Kim"]
    for name in test_names:
        predict(trained_model, name, all_categories, device)