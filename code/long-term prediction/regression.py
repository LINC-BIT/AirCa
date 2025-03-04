import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from glob import glob

# ----------------------------
# 全局常量，或者也可以改成命令行参数
# ----------------------------
DATA_DIR = "G:\\loading_benchmark\\bakFlightLoadData\\bakFlightLoadData"  # 数据文件夹路径
AIRCRAFT_MODELS = ["A320", "B737", "A330", "A350", "B777", "B787"]
TRAIN_DAYS = 40
TEST_DAYS = 47

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

# --------------------------------
# 注意：当前仅用一层线性映射做预测
# 如果要真正使用多层 LSTM，可自行替换为：
#
# class LSTMPredictor(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTMPredictor, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)  # out: [batch_size, seq_len, hidden_size]
#         # 在这里根据需要输出 out[:, -1, :] 或者整个序列
#         out = self.fc(out)
#         return out
# --------------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMPredictor, self).__init__()
        # 这里虽然传入了 hidden_size, num_layers，但暂未真正使用
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.fc(x)  # 仅线性
        return out

def train_model(train_loader, model, criterion, optimizer, max_epochs=50):
    """
    您原先的代码使用了一个 while True 的循环，并在达到 acc>0.5 后中断；
    同时这里也结合了一个 max_epochs 的上限，防止无限循环。
    """
    model.train()
    epoch = 0
    while True:
        epoch += 1
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            # batch: [batch_size, 1] -> 需要加一个维度变成 [batch_size, seq_len=1, features=1]
            batch = batch.unsqueeze(1)
            predictions = model(batch)
            loss = criterion(predictions, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 每个 epoch 后，在训练集上测试一下准确率（acc>0.5）
        _, _, _, _, acc = test_model(train_loader, model, batch_size=train_loader.batch_size)

        # 每 50 个 epoch 打印一次 acc
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Acc on train: {acc:.3f}")

        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

        # 如果达到条件则停止
        if acc > 0.5:
            print(f"Early stopping triggered at epoch={epoch}, Acc={acc:.3f}")
            break

        # 如果超过用户指定的最大轮数，也要停止
        if epoch >= max_epochs:
            print("Reached max_epochs, stopping training.")
            break

def test_model(test_loader, model, batch_size):
    """
    batch_size：用来区分测试时是 batch_size=1 还是其他；
    当前逻辑：如果 batch_size=1，则认为是一条一条计算 mape；
             如果 batch_size>1，则在循环里逐元素判断。
    """
    model.eval()
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.unsqueeze(1)  # [batch_size, 1, 1]
            preds = model(batch)
            predictions.append(preds.squeeze().numpy())
            true_values.append(batch.squeeze().numpy())

    acc = 0
    mae_list = []
    mape_list = []

    # 逐样本计算 MAE 和 MAPE
    for i in range(len(true_values)):
        mae = np.abs(predictions[i] - true_values[i])
        # 避免除 0
        denom = true_values[i] if true_values[i] != 0 else 1e-9
        mape = np.abs(predictions[i] - true_values[i]) / denom

        mae_list.append(mae)
        mape_list.append(mape)

        if batch_size == 1:
            # 单样本场景
            if mape * 100 < 7:
                acc += 1
        else:
            # 多样本场景
            count_local = 0
            for j in range(len(mape)):
                if mape[j] * 100 < 7:
                    count_local += 1
            acc += count_local  # 统计所有样本中 MAP < 7% 的个数

    total_samples = len(true_values) * batch_size
    accuracy = acc / total_samples if total_samples > 0 else 0.0

    return true_values, predictions, mae_list, mape_list, accuracy

def main(args):
    # 加载数据
    data_by_day = load_and_process_data(DATA_DIR, AIRCRAFT_MODELS)

    # 按机型分开训练和测试
    for model_name in AIRCRAFT_MODELS:
        print(f"Processing model: {model_name}")
        fw = open(f'G:\\loading_benchmark\\result\\pred_{model_name}_reg1', 'w')

        # 准备训练和测试数据
        train_processed, test_processed = prepare_data(data_by_day, TRAIN_DAYS, TEST_DAYS)
        train_data = train_processed[model_name]
        test_data = test_processed[model_name]

        if not train_data or not test_data:
            print(f"No data available for {model_name}")
            fw.close()
            continue

        # 构建 DataLoader
        train_dataset = AircraftDataset(train_data)
        test_dataset = AircraftDataset(test_data)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 定义模型
        input_size = 1   # 仅平均载货重量作为输入
        hidden_size = 64 # （当前没有用到）
        num_layers = args.num_layers
        output_size = 1
        model_instance = LSTMPredictor(input_size, hidden_size, num_layers, output_size)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model_instance.parameters(), lr=args.lr)

        # 训练模型（带 early stopping 逻辑）
        train_model(train_loader, model_instance, criterion, optimizer, max_epochs=args.epochs)

        # 测试模型
        true_vals, preds, mae_list, mape_list, acc = test_model(test_loader, model_instance, batch_size=1)
        for i in range(len(preds)):
            fw.write(
                f"{true_vals[i]}\t{preds[i]}\t{mae_list[i]}\t{mape_list[i]}\t{acc}\n"
            )
        fw.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a regression model (demo) for predicting aircraft load.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers (currently unused in example)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--epochs", type=int, default=200, help="Max number of training epochs")
    args = parser.parse_args()

    main(args)
