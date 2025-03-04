import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from glob import glob

# 参数设置（可固定，或进一步改成命令行参数）
DATA_DIR = "G:\\loading_benchmark\\bakFlightLoadData\\bakFlightLoadData"  # 数据文件夹路径
AIRCRAFT_MODELS = ["A320", "B737", "A330", "A350", "B777", "B787"]
TRAIN_DAYS = 40
TEST_DAYS = 47

def load_and_process_data(data_dir, aircraft_models):
    """
    加载并处理原始数据，将每天的同一机型的载货重量平均值存储在列表中。
    """
    all_files = sorted(glob(os.path.join(data_dir, "BAKFLGITH_LOADDATA*.csv")))
    data_by_day = []

    for file in all_files:
        df = pd.read_csv(
            file,
            header=None,
            usecols=[0, 1, 3],
            names=["fid", "fleetId", "weight"],
            encoding='ISO-8859-1'
        )
        daily_data = {}
        for model in aircraft_models:
            model_data = df[df["fleetId"].str.contains(model, na=False)]
            if not model_data.empty:
                daily_avg = model_data["weight"].mean()  # 每天的平均载货重量
                daily_data[model] = daily_avg
        data_by_day.append(daily_data)

    return data_by_day

def prepare_data(data_by_day, train_days, test_days):
    """
    准备按机型的训练和测试数据，每天计算平均值，分为训练和测试集。
    """
    train_data = data_by_day[:train_days]
    test_data = data_by_day[:train_days + test_days]

    train_processed = {model: [] for model in AIRCRAFT_MODELS}
    test_processed = {model: [] for model in AIRCRAFT_MODELS}

    for daily_data in train_data:
        for model in AIRCRAFT_MODELS:
            if model in daily_data:
                train_processed[model].append(daily_data[model])

    for daily_data in test_data:
        for model in AIRCRAFT_MODELS:
            if model in daily_data:
                test_processed[model].append(daily_data[model])

    return train_processed, test_processed


# 自定义数据集
class AircraftDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor([self.data[idx]], dtype=torch.float32)


# CNN 模型，支持可变的卷积层数
class CNNSPredictor(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2):
        """
        这里简单实现了一个可根据num_layers决定多少层Conv1d的结构。
        为了演示，最多写了3层，若需要更多层可以进一步扩展。
        """
        super(CNNSPredictor, self).__init__()
        self.num_layers = num_layers

        # 第一层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1, padding=1)
        # 第二层
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, padding=1)
        # 第三层（可选）
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=1, padding=1)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=0.01)

        # 根据卷积层数来决定全连接层输入维度
        if self.num_layers == 1:
            fc_in_dim = 64
        elif self.num_layers == 2:
            fc_in_dim = 32
        else:  # num_layers >= 3
            fc_in_dim = 16

        self.fc1 = nn.Linear(fc_in_dim, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        # x: [batch_size, 1, seq_len]
        # 第一层
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dropout(x)

        # 第二层
        if self.num_layers >= 2:
            x = self.conv2(x)
            x = self.pool(x)
            x = self.dropout(x)

        # 第三层
        if self.num_layers >= 3:
            x = self.conv3(x)
            x = self.pool(x)
            x = self.dropout(x)

        # 展平
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# 训练和测试函数
def train_model(train_loader, model, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            # [batch_size, 1, seq_len]，因为只输入一个特征，这里seq_len=1
            batch = batch.unsqueeze(1)
            predictions = model(batch)
            loss = criterion(predictions, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


def test_model(test_loader, model):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.unsqueeze(1)
            preds = model(batch)
            predictions.append(preds.squeeze().numpy())
            true_values.append(batch.squeeze().numpy())

    mae_list = []
    mape_list = []
    acc = 0
    for i in range(len(true_values)):
        mae = np.abs(predictions[i] - true_values[i])  # 平均绝对误差
        mape = np.abs(predictions[i] - true_values[i]) / (true_values[i] + 1e-9)  # 防止除0
        mae_list.append(mae)
        mape_list.append(mape)
        # 这里的“7%以内”可根据需求自行调整
        if mape * 100 < 7:
            acc += 1

    # 这里返回整体平均或每个样本的统计均可
    return true_values, predictions, mae_list, mape_list, acc / len(true_values)


def main(args):
    # 加载数据
    data_by_day = load_and_process_data(DATA_DIR, AIRCRAFT_MODELS)

    # 按机型分开训练和测试
    for model_name in AIRCRAFT_MODELS:
        print(f"Processing model: {model_name}")
        fw = open(f'G:\\loading_benchmark\\result\\pred_{model_name}_cnn2', 'w', encoding='utf-8')

        # 准备训练和测试数据
        train_processed, test_processed = prepare_data(data_by_day, TRAIN_DAYS, TEST_DAYS)
        train_data = train_processed[model_name]
        test_data = test_processed[model_name]

        if not train_data or not test_data:
            print(f"No data available for {model_name}")
            fw.close()
            continue

        # 创建Dataset和DataLoader
        train_dataset = AircraftDataset(train_data)
        test_dataset = AircraftDataset(test_data)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 定义模型
        input_size = 1   # 仅平均载货重量作为输入
        output_size = 1  # 预测平均载货重量
        model_instance = CNNSPredictor(input_size, output_size, num_layers=args.num_layers)

        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=args.lr)

        # 训练模型
        train_model(train_loader, model_instance, criterion, optimizer, epochs=args.epochs)

        # 测试模型
        true_vals, preds, mae_list, mape_list, acc = test_model(test_loader, model_instance)

        # 写入文件：true, pred, mae, mape, acc
        for i in range(len(preds)):
            fw.write(
                f"{true_vals[i]}\t{preds[i]}\t{mae_list[i]}\t{mape_list[i]}\t{acc}\n"
            )
        fw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN model for predicting aircraft load.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of Conv1d layers in the CNN")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=600, help="Number of training epochs")
    args = parser.parse_args()

    main(args)
