import torch
import torch.nn as nn
import torch.nn.functional as F

# 自注意蒸馏
class SelfAttentionDistil(nn.Module):
    def __init__(self, c_in):
        super(SelfAttentionDistil, self).__init__()
        self.conv = nn.Conv1d(c_in, c_in, kernel_size=3, padding=2, padding_mode="circular")  # c_in 输入通道数，输出通道数。
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = torch.transpose(x, 1, 2)
        return x

# 一个标准的 Transformer 编码器层，包含自注意力机制、前馈神经网络、残差连接、层归一化和 dropout
class EncoderLayer(nn.Module):
    def __init__(self, attention, embed_dim, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * embed_dim  # 4*d_model is used in the paper，前向传播的维度
        self.attention = attention  # 多头注意力机制
        self.conv1 = nn.Conv1d(embed_dim, d_ff, kernel_size=1)  # 前馈神经网络由两个一维卷积层组成
        self.conv2 = nn.Conv1d(d_ff, embed_dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(embed_dim)   # 层归一化
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attention_mask=None):
        new_x, attention = self.attention(x, x, x, attention_mask=attention_mask)
        # new_x 是通过多头注意力计算得到的输出，attention
        x = x + self.dropout(new_x)  # 残差连接和dropout (8,197,768)

        # 前馈神经网络
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attention  # 输出和注意力 ，第二次归一化，加上残差连接


# 包含多个注意力层和卷积层的编码器模块
class Encoder(nn.Module):
    def __init__(self, attention_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attention_layers = nn.ModuleList(attention_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None  # 将 attention_layers 和 conv_layers 转换为 nn.ModuleList
        self.norm = norm_layer

    def forward(self, x, attention_mask=None):
        attentions = []
        if self.conv_layers is not None:  # 如果有卷积层
            for attention_layer, conv_layer in zip(self.attention_layers, self.conv_layers):
                x, attention = attention_layer(x, attention_mask=attention_mask)  # 计算注意力输出和注意力权重
                x = conv_layer(x)  # 卷积层
                attentions.append(attention)
            x, attention = self.attention_layers[-1](x)  # 最后一层的注意力输出和注意力权重
            attentions.append(attention)
        else:
            for attention_layer in self.attention_layers:
                x, attention = attention_layer(x, attention_mask=attention_mask)
                attentions.append(attention)
        if self.norm is not None:
            x = self.norm(x)
        return x, attentions


# 将多个编码器（Encoder）层组合在一起，形成一个编码器堆栈
class EncoderStack(nn.Module):
    def __init__(self, encoders):
        super(EncoderStack).__init__()
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x, attention_mask=None):
        inp_len = x.size(1)  # 输入序列长度 ，
        x_stack = []  # 保存各个编码器的输出
        attentions = []  # 保存各个编码器的注意力权重
        for encoder in self.encoders:
            if encoder is None:  # 如果编码器为空，则跳过
                inp_len //= 2  # 输入序列长度减半
                continue
            x, attention = encoder(x[:, -inp_len:, :])  # 编码器存在，则计算输出和注意力权重
            x_stack.append(x)
            attentions.append(attention)
            inp_len //= 2
        x_stack = torch.cat(x_stack, -2)  # 连接各个编码器的输出
        return x_stack, attentions
