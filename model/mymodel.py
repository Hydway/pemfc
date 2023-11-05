import torch
import torch.nn as nn
from torch.nn import Dropout
import torch.nn.functional as F
from iTransformer import iTransformer



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


class Stem(nn.Module):
    def __init__(self, win_size, inchanels=1, outchanels=1, groups=1, kernel=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=inchanels,
                               out_channels=outchanels,
                               kernel_size=kernel,
                               padding=1,
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=inchanels,
                               out_channels=outchanels,
                               kernel_size=kernel,
                               padding=1,
                               stride=1)
        self.conv3 = nn.Conv2d(in_channels=inchanels,
                               out_channels=outchanels,
                               kernel_size=kernel,
                               padding=1,
                               stride=1)

        self.pointwise = nn.Conv2d(in_channels=win_size, out_channels=win_size, kernel_size=1)

        self.proj = nn.Sequential(
            self.conv1,
            # nn.BatchNorm2d(outchanels),
            nn.GELU(),
            self.conv2,
            # nn.BatchNorm2d(int(hidden_dim * 2)),
            nn.GELU(),
            self.conv3,
            # nn.BatchNorm2d(int(hidden_dim * 4)),
            # nn.GELU(),
        )

    def forward(self, x):
        tensor_in = x
        B, H, W = x.size()
        x = x.unsqueeze(1)
        x = self.proj(x)
        # x = x.view([B, H, W])
        x = x.view(B, H, 1, W)
        x = self.pointwise(x)
        x = x.squeeze()
        return x + tensor_in


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


# 可学习的绝对位置编码
class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(AbsolutePositionalEncoding, self).__init__()
        self.pos_encoding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1)).unsqueeze(0).to(x.device)
        return x + self.pos_encoding(positions)


# 可学习的相对位置编码
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(RelativePositionalEncoding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(2*max_len+1, d_model)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        col_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(seq_len, 1)
        row_indices = torch.arange(seq_len, device=x.device).unsqueeze(1).repeat(1, seq_len)
        distance = col_indices - row_indices
        distance = distance + seq_len
        embeddings = self.embed(distance)
        return x + embeddings



class cpe_pos_embb(nn.Module):
    def __init__(self, d_model):
        super(cpe_pos_embb, self).__init__()
        self.pos_embed = nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)
        pass

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = x + self.pos_embed(x)
        x = x.permute(1, 2, 0)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, head_num, drop=0., mlp_ratio=4, win_size=100):
        super(TransformerBlock, self).__init__()
        self.ori_d = 8

        # 位置编码
        # self.pos_embed = cpe_pos_embb(d_model)
        # self.pos_embed = PositionalEncoding(d_model)
        # self.pos_embed = AbsolutePositionalEncoding(d_model)
        self.pos_embed = RelativePositionalEncoding(d_model)

        self.att = nn.MultiheadAttention(embed_dim=d_model, num_heads=head_num)
        self.b_norm = nn.BatchNorm1d(self.ori_d)
        self.nn = nn.Linear(d_model, self.ori_d)
        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.norm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        self.mlp_hidden_dim = int(self.ori_d * mlp_ratio)
        self.mlp = Mlp(in_features=d_model, hidden_features=self.mlp_hidden_dim, out_features=d_model,
                       act_layer=nn.GELU, drop=drop)

    def forward(self, x, mask=None):
        # Multi-head attention
        x = self.pos_embed(x)
        attn_output, _ = self.att(x, x, x, attn_mask=mask)
        # x = self.norm(self.nn(x + attn_output))
        x = self.norm(x + attn_output)
        x = self.drop(x)
        x = x.transpose(1, 2)
        # x = self.b_norm(x)
        x = x.transpose(1, 2)
        x = self.act(x)

        # Add & Norm
        x_mlp = self.mlp(self.norm1(x))
        x = self.norm2(x_mlp + x)

        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=256, drop=0, win_size=100):
        super(TransformerModel, self).__init__()
        self.stem1 = Stem(win_size=win_size)
        self.stem2 = Stem(win_size=win_size)
        self.stem3 = Stem(win_size=win_size)
        hidden_features = input_dim * 4
        self.fc_in1 = Mlp(input_dim, hidden_features, out_features=d_model)
        self.fc_in2 = Mlp(input_dim, hidden_features, out_features=d_model)
        self.fc_in3 = Mlp(input_dim, hidden_features, out_features=d_model)
        self.neck1 = Mlp(d_model, hidden_features, input_dim)
        self.neck2 = Mlp(d_model, hidden_features, input_dim)
        # self.fc_in = nn.Linear(input_dim, d_model)
        self.b_norm1 = nn.BatchNorm1d(win_size)
        self.b_norm2 = nn.BatchNorm1d(win_size)
        self.b_norm3 = nn.BatchNorm1d(win_size)
        # self.positional_enc = PositionalEncoding(d_model)
        self.iTransformerBlock = iTransformer(
            # num_variates = ,
            # lookback_len = ,
            # depth = ,
            # dim = ,
        )
        self.block1 = TransformerBlock(d_model=d_model, head_num=16, drop=drop, win_size=win_size)
        self.block2 = TransformerBlock(d_model=d_model, head_num=16, drop=drop, win_size=win_size)
        self.block3 = TransformerBlock(d_model=d_model, head_num=16, drop=drop, win_size=win_size)
        self.block4 = TransformerBlock(d_model=d_model, head_num=16, drop=drop, win_size=win_size)
        self.block5 = TransformerBlock(d_model=d_model, head_num=8, drop=drop, win_size=win_size)
        self.block6 = TransformerBlock(d_model=d_model, head_num=8, drop=drop, win_size=win_size)
        self.block7 = TransformerBlock(d_model=d_model, head_num=8, drop=drop, win_size=win_size)
        self.block8 = TransformerBlock(d_model=d_model, head_num=8, drop=drop, win_size=win_size)
        self.block9 = TransformerBlock(d_model=d_model, head_num=16, drop=drop, win_size=win_size)
        self.block10 = TransformerBlock(d_model=d_model, head_num=16, drop=drop, win_size=win_size)
        self.fc_out1 = nn.Linear(d_model, int(d_model / 2))
        self.fc_out2 = nn.Linear(int(d_model / 4), 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(int(d_model / 4))
        self.act1 = nn.Tanh()
        self.act2 = nn.Tanh()
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        x = self.stem1(x)
        x = self.fc_in1(x)
        # x = self.b_norm1(x)
        # x = F.leaky_relu(x)
        x = self.block1(x)
        # x = self.block2(x)
        # x = self.block3(x)
        # x = self.block4(x)

        # x = self.neck1(x)
        #
        # x = self.stem2(x)
        # x = self.fc_in2(x)
        # x = self.b_norm2(x)
        # # x = F.leaky_relu(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        # x = self.block8(x)

        # x = self.neck2(x)
        #
        # x = self.stem3(x)
        # x = self.fc_in3(x)
        # # x = self.b_norm3(x)
        # # x = F.leaky_relu(x)
        # x = self.block9(x)
        # x = self.block10(x)

        x = x[:, -1, :]
        x = self.fc_out1(x)
        # x = self.act1(x)
        x = self.avg_pool(x)
        # x = self.drop(x)
        x = self.fc_out2(x)
        # x = self.act2(x)
        # x = self.drop(x)
        return x