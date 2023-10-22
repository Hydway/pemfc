import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设定设备为GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

window_size = 10
BATCH_SIZE = 15
num_epochs = 20
lr=1e-4

data_dynamic = pd.read_csv('./data/quasi_1000_MAF_data.csv')
quasi_features = data_dynamic.iloc[0:5005+window_size, [0, 2, 3, 4, 5, 6, 7, 8, 9]].values
totalvolt_quasi_data = data_dynamic.iloc[0:5005+window_size, [1]].values

scaler_features = MinMaxScaler()
quasi_features = scaler_features.fit_transform(quasi_features)

scaler_voltage = MinMaxScaler()
totalvolt_quasi_data = scaler_voltage.fit_transform(totalvolt_quasi_data)

X_seq, y_seq = [], []

for i in range(window_size, len(quasi_features)):
    X_seq.append(quasi_features[i-window_size:i, :])
    y_seq.append(totalvolt_quasi_data[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# 数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self._positional_encoding(d_model, max_len)

    def _get_angles(self, position, d_model):
        angles = 1 / torch.pow(10000, (2 * (torch.arange(d_model)[None, :] // 2)) / d_model)
        return position * angles

    def _positional_encoding(self, d_model, max_len):
        angle_rads = self._get_angles(torch.arange(max_len)[:, None], d_model)
        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
        return angle_rads[None, ...]

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1), :].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=512):
        super(TransformerModel, self).__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        self.positional_enc = PositionalEncoding(d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc_in(x))
        x = self.positional_enc(x)
        x, _ = self.attention(x, x, x)
        x = x[:, -1, :]
        x = self.fc_out(x)
        return x

model = TransformerModel(input_dim=X_train_tensor.shape[-1]).to(device)
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
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss/len(train_loader))

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
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


def create_sequences(data, window_size):
    """
    Creates sequences from the data.

    Parameters:
    - data: Original data to be converted into sequences.
    - window_size: Size of the window for the sequences.

    Returns:
    - Sequences array.
    """
    sequences = []
    for i in range(len(data) - window_size):
        sequence = data[i:(i + window_size)]
        sequences.append(sequence)
    return np.array(sequences)

# 创建整个数据集的时间序列
quasi_features_1000 = scaler_features.transform(data_dynamic.iloc[:, [0, 2, 3, 4, 5, 6, 7, 8, 9]].values)
quasi_features_1000_sequences = torch.FloatTensor(create_sequences(quasi_features_1000, window_size)).to(device)

# 使用训练好的模型进行预测
"""
dataset = TensorDataset(quasi_features_1000_sequences)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model.eval()
y_pred_1000 = []

with torch.no_grad():
    
    for batch in data_loader:
        # 将数据移到 GPU 上
        batch = batch[0].to(device)  # 假设 device 是你的 GPU
        
        # 预测
        batch_pred = model(batch)
        
        # 将预测结果移到 CPU 上，并转化为 numpy 数组
        y_pred_1000.append(batch_pred.cpu().numpy())
        
# 将所有批次的预测结果汇总
y_pred_1000 = np.concatenate(y_pred_1000)
"""

model.eval()
with torch.no_grad():
    y_pred_1000 = model(quasi_features_1000_sequences).cpu().numpy()

# 反归一化预测数据
y_pred_1000_rescaled = scaler_voltage.inverse_transform(y_pred_1000)

# 绘制真实的和预测的 U_{tot} 值
plt.figure(figsize=(12, 6))
plt.plot(data_dynamic.iloc[window_size:5006+window_size, 1].values, label='True Value', linewidth=2, color='blue')
plt.plot(y_pred_1000_rescaled[:5006], label='Predicted Value', linestyle='--', color='red')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Voltage (Utot)', fontsize=14)
plt.title('True vs Predicted Voltages for the first 5016 data points', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.grid(True)
plt.savefig('Pred.png')  # 保存图像到文件

# 计算评价指标
y_true_5016 = data_dynamic.iloc[window_size:5005+window_size, 1].values
mse = mean_squared_error(y_true_5016, y_pred_1000_rescaled[:5005])
rmse = np.sqrt(mse)
r2 = r2_score(y_true_5016, y_pred_1000_rescaled[:5005])

print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")
