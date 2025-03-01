import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from glob import glob

# 参数设置
DATA_DIR = "G:\\loading_benchmark\\bakFlightLoadData\\bakFlightLoadData"  # 数据文件夹路径
AIRCRAFT_MODELS = ["A320", "B737", "A330", "A350", "B777", "B787"]
TRAIN_DAYS = 40
TEST_DAYS = 47

# 数据处理
def load_and_process_data(data_dir, aircraft_models):
    all_files = sorted(glob(os.path.join(data_dir, "BAKFLGITH_LOADDATA*.csv")))
    data_by_day = []

    for file in all_files:
        df = pd.read_csv(file, header=None, usecols=[0, 1, 3], names=["fid", "fleetId", "weight"],
                         encoding='ISO-8859-1')
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


# CNN 模型
class CNNSPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNSPredictor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=0.01)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, padding=1)
        self.fc1 = nn.Linear(32, 64)  # 全连接层，用于输出之前的隐藏层
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        # 输入 x 形状: [batch_size, 1, seq_len]
        x = self.conv1(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)  # 展平
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
            batch = batch.unsqueeze(1)  # [batch, 1, seq_len] 扩展维度，适应 Conv1d
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
            batch = batch.unsqueeze(1)  # [batch_size, 1, seq_len]
            preds = model(batch)
            predictions.append(preds.squeeze().numpy())
            true_values.append(batch.squeeze().numpy())

    mae_list = []
    mape_list = []
    acc = 0
    for i in range(len(true_values)):
        mae = np.abs(predictions[i] - true_values[i])  # 平均绝对误差
        mape = np.abs(predictions[i] - true_values[i])/true_values[i]
        mae_list.append(mae)
        mape_list.append(mape)
        if mape * 100 < 7:
            acc += 1
    return true_values, predictions, mae_list, mape_list, acc / len(true_values)


# 主程序
if __name__ == "__main__":
    # 加载数据
    data_by_day = load_and_process_data(DATA_DIR, AIRCRAFT_MODELS)

    # 按机型分开训练和测试
    results = {}
    for model in AIRCRAFT_MODELS:
        print(f"Processing model: {model}")
        fw = open(f'G:\\loading_benchmark\\result\\pred_{model}_cnn2', 'w')
        # 准备训练和测试数据
        train_processed, test_processed = prepare_data(data_by_day, TRAIN_DAYS, TEST_DAYS)
        train_data = train_processed[model]
        test_data = test_processed[model]

        if not train_data or not test_data:
            print(f"No data available for {model}")
            continue

        train_dataset = AircraftDataset(train_data)
        test_dataset = AircraftDataset(test_data)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 定义模型和训练
        input_size = 1  # 仅平均载货重量作为输入
        output_size = 1  # 预测平均载货重量
        model_instance = CNNSPredictor(input_size, output_size)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=0.001)

        # 训练模型
        train_model(train_loader, model_instance, criterion, optimizer, epochs=600)

        # 测试模型
        true, predictions, mae, mse, acc = test_model(test_loader, model_instance)
        for i in range(len(predictions)):
            fw.write(f"{true[i]}\t{predictions[i]}\t{mae[i]}\t{mse[i]}\t{acc}\n")

    # 输出结果
    # for model, preds in results.items():
    #     print(f"Results for {model}:")
    #     print(preds)
