import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer

from model.iTransformer import iTransformer

import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

from utils import data_clean

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

matplotlib.use('Agg')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
seed_everything(seed)

# 设定设备为GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

window_size = 500
BATCH_SIZE  = 512
depth_num = 4
head = 8
d_attn = window_size
Drop = 0.2
num_epochs  = 10
lr = 1e-4
pred_step = window_size//1  # 可以根据自己的需求调整

# 读取train / pred原始数据
train_data = pd.read_csv('./data/FC2_Ageing_part1.csv', encoding='unicode_escape')
pred_data  = pd.read_csv('./data/FC2_Ageing_part2.csv', encoding='unicode_escape')
# print("raw train data: ", train_data.head())
# print("raw pred data: ", pred_data.head())
# print("len raw data:", len(train_data))

# 异常值处理
train_data = data_clean(train_data, process_data=False)
# train_data.drop(train_data.tail(100).index, inplace=True)
pred_data  = data_clean(pred_data, process_data=False)

# train数据预处理
# train_quasi_features = train_data.drop(train_data.columns[[0, 6, -1]], axis=1).values
train_quasi_features = train_data.drop(train_data.columns[[0, ]], axis=1).values
train_quasi_vol = train_data.iloc[:, [-1]].values
# print("dropped data: ", train_quasi_features[0])
# print("len dropped data:", len(train_quasi_features[0]))

# pred数据预处理
pred_quasi_features = pred_data.drop(pred_data.columns[[0, ]], axis=1).values
# pred_quasi_features = pred_data.drop(pred_data.columns[[0, 6, ]], axis=1).values
pred_quasi_vol = pred_data.iloc[:, [-1]].values
# print("dropped data: ", pred_quasi_features[0])
# print("len dropped data:", len(pred_quasi_features[0]))


# 归一化特征
scaler_features = StandardScaler()
train_quasi_features = scaler_features.fit_transform(train_quasi_features)
pred_quasi_features = scaler_features.transform(pred_quasi_features)

# 归一化电压数据
scaler_voltage = StandardScaler()
train_quasi_vol = scaler_voltage.fit_transform(train_quasi_vol)

# pred_quasi_vol = scaler_voltage.fit_transform(pred_quasi_vol)

# 准备数据结构
X_train_seq, y_train_seq = [], []
X_pred_seq, y_pred_seq, y_pred_ori_seq = [], [], []

for i in range(window_size, len(train_quasi_features) - pred_step):
    X_train_seq.append(train_quasi_features[i - window_size:i, :])
    y_train_seq.append(train_quasi_vol[i:i + pred_step])

for i in range(window_size, len(pred_quasi_features) - pred_step):
    X_pred_seq.append(pred_quasi_features[i - window_size:i, :])
    y_pred_seq.append(pred_quasi_vol[i:i + pred_step])
    y_pred_ori_seq.append(pred_quasi_vol[i])

print(len(X_train_seq))
X_train_seq = np.array(X_train_seq)
y_train_seq = np.array(y_train_seq)
print(y_train_seq.shape)

print(len(X_pred_seq))
print(len(y_pred_seq))
print(len(y_pred_ori_seq))
X_pred_seq = np.array(X_pred_seq)
y_pred_seq = np.array(y_pred_seq)


# print("train x: ", X_train_seq)
# print("train y", y_train_seq)

# 划分训练集验证集
X_train, X_test, y_train, y_test = train_test_split(X_train_seq, y_train_seq, test_size=0.2, random_state=seed)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).squeeze(2).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).squeeze(2).to(device)


# 数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型
model = iTransformer(
    num_variates=X_train_tensor.shape[-1],
    lookback_len=window_size,
    dim=d_attn,
    depth=depth_num,
    heads=head,
    dim_head=d_attn,
    pred_length=(pred_step,),
    num_tokens_per_variate=1,
    use_reversible_instance_norm=True,
    flash_attn = False,
    attn_dropout = Drop,
).to(device)

print(model)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        outputs = preds[pred_step][..., -1]
        # print(preds[pred_step].size())
        # print(outputs.size())
        # print(y_batch.size())
        # print(outputs)
        # print(y_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss/len(train_loader))

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            outputs = preds[pred_step][..., -1]   # Fetch the output for the prediction length 1
            loss = criterion(outputs, y_batch)
            running_loss += loss.item()
    test_losses.append(running_loss/len(test_loader))

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")


# 绘制训练和验证的MSE
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training MSE')
plt.plot(test_losses, label='Validation MSE')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend(loc='upper right')
plt.title("Training and Validation MSE over Epochs")
plt.savefig('MSE.png')  # 保存图像到文件


def predict(X_pred_seq, pred_step):
    # 创建整个数据集的时间序列
    X_pred_seq = torch.FloatTensor(X_pred_seq).to(device)

    # 使用训练好的模型进行预测
    dataset = TensorDataset(X_pred_seq)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    y_pred = []
    i_list = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # 每次只预测一个序列，然后跳过pred_step个序列
            if i % pred_step != 0:
                continue

            # 将数据移到 GPU 上
            batch = batch[0].to(device)

            # 预测
            preds = model(batch)
            batch_pred = preds[pred_step][..., -1]  # Fetch the output for the prediction length 1

            # 将预测结果移到 CPU 上，并转化为 numpy 数组
            i_list.append(i)
            y_pred.append(batch_pred.cpu().numpy())

    # 将所有批次的预测结果汇总
    y_pred = np.concatenate(y_pred)

    return y_pred, i_list


# y_valid = predict(X_train_seq)
print("X_train_seq.shape:", X_train_seq.shape)
y_valid_, _ = predict(X_train_seq, pred_step)
print("y_valid_.shape:", y_valid_.shape)
y_valid_ = y_valid_.reshape(-1, 1)
y_valid = scaler_voltage.inverse_transform(y_valid_)
print("y_valid.shape:", y_valid.shape)


y_pred_1000_rescaled_, i_list = predict(X_pred_seq, pred_step)
y_pred_1000_rescaled_ = y_pred_1000_rescaled_.reshape(-1, 1)[:len(X_pred_seq)]
print("X_pred_seq.shape:", X_pred_seq.shape)
print("len i_list", len(i_list))
y_pred_1000_rescaled  = scaler_voltage.inverse_transform(y_pred_1000_rescaled_)

print("y_pred_1000_rescaled_.shape:", y_pred_1000_rescaled_.shape)
print("y_pred_1000_rescaled.shape:", y_pred_1000_rescaled.shape)


# 绘制真实的和预测的 U_{tot} 值
plt.figure(figsize=(12, 6))
plt.plot(train_data.iloc[window_size:, -1].values, label='True Value', linewidth=2, color='blue')
plt.plot(y_valid, label='Predicted Value', linestyle='--', color='red')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Voltage (Utot)', fontsize=14)
plt.title('True vs Predicted Voltages for the first 5016 data points', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.grid(True)
plt.savefig('Valid.png')  # 保存图像到文件


# 绘制真实的和预测的 U_{tot} 值
plt.figure(figsize=(12, 6))
plt.plot(pred_data.iloc[window_size:, -1].values, label='True Value', linewidth=2, color='blue')
plt.plot(y_pred_1000_rescaled, label='Predicted Value', linestyle='--', color='red')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Voltage (Utot)', fontsize=14)
plt.title('True vs Predicted Voltages for the first 5016 data points', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.grid(True)
plt.savefig('Pred.png')  # 保存图像到文件

# 计算评价指标
mse = mean_squared_error(y_pred_ori_seq, y_pred_1000_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_pred_ori_seq, y_pred_1000_rescaled)

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
