import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, head_num, drop=0., mlp_ratio=4):
        super(TransformerBlock, self).__init__()
        self.pos_embed = nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)
        self.att = nn.MultiheadAttention(embed_dim=d_model, num_heads=head_num)
        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = Mlp(in_features=d_model, hidden_features=self.mlp_hidden_dim,
                       act_layer=nn.GELU, drop=drop)

    def forward(self, x, mask=None):
        # Multi-head attention
        x = x.permute(2, 0, 1)
        x = x + self.pos_embed(x)
        x = x.permute(1, 2, 0)
        # x = self.norm(x)
        attn_output, _ = self.att(x, x, x, attn_mask=mask)
        # Add & Norm
        x = self.mlp(self.norm1(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=256):
        super(TransformerModel, self).__init__()
        self.fc_in = nn.Linear(input_dim, d_model)
        # self.positional_enc = PositionalEncoding(d_model)
        self.block1 = TransformerBlock(d_model=d_model, head_num=8, )
        self.block2 = TransformerBlock(d_model=d_model, head_num=8, )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc_in(x))
        x = self.block1(x)
        x = self.block2(x)
        x = x[:, -1, :]
        x = self.fc_out(x)
        return x