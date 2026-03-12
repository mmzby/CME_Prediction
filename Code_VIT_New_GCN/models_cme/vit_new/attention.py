import torch
import torch.nn as nn
import numpy as np
import math
from models_cme.vit_new.masking import triangular_causal_mask, prob_mask
# from masking import triangular_causal_mask, prob_mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, **kwargs):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attention_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attention_mask is None:
                attention_mask = triangular_causal_mask(B, L, device=queries.device)
            scores.masked_fill_(attention_mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class ProbSparseAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(ProbSparseAttention, self).__init__()
        self.mask_flag = mask_flag  # 是否使用掩码
        self.factor = factor  # 用于计算样本数量的因子
        self.scale = scale  # 注意力分数的缩放因子
        self.output_attention = output_attention  # 是否输出注意力矩阵
        self.dropout = nn.Dropout(attention_dropout)  # 注意力矩阵的dropout层

    def forward(self, queries, keys, values, attention_mask):
        B, L_Q, H, D = queries.shape  # B: batch_size, L_Q: query序列长度, H: 多头注意力头数, D: 输入维度
        _, L_K, _, _ = keys.shape  # L_K: key序列长度

        queries = torch.transpose(queries, 2, 1)  # B, H, L_Q, D
        keys = torch.transpose(keys, 2, 1)
        values = torch.transpose(values, 2, 1)

        U_part = int(self.factor * math.ceil(math.log(L_K)))  # c * ln(L_K) ，
        u = int(self.factor * math.ceil(math.log(L_Q)))  # c * ln(L_Q)， U_part和u是用于计算top_k的采样数量

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)    # 调用_prob_QK函数，得到scores_top和index

        scale = self.scale or 1.0 / math.sqrt(D)  # 缩放因子
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)  # 获取初始上下文
        # update the context with selected top_k queries
        context, attention = self._update_context(context, values, scores_top, index, L_Q, attention_mask)

        return context.transpose(2, 1).contiguous(), attention

    # 通过扩展 keys，随机选择 sample_k 并计算 Q 和 K 的内积。
    # 计算每个 Q 的稀疏性指标，并找到前 n_top 个重要的索引
    def _prob_QK(self, queries, keys, sample_k, n_top):
        B, H, L_K, E = keys.shape
        _, _, L_Q, _ = queries.shape

        # calculate the sampled Q_K
        K_expand = keys.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor * ln(L_K)) * L_Q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = (queries.unsqueeze(-2) @ K_sample.transpose(-2, -1)).squeeze()

        # find the top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = queries[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]  # factor * ln(L_Q)
        Q_K = Q_reduce @ keys.transpose(-2, -1)  # factor * ln(L_Q) * L_K

        return Q_K, M_top

    # 根据是否使用掩码来计算初始上下文。对于无掩码的情况，使用 values 的平均值；若使用掩码，则 L_Q 必须等于 L_V
    def _get_initial_context(self, values, L_Q):
        B, H, L_V, D = values.shape
        if not self.mask_flag:
            V_mean = values.mean(dim=-2)
            context = V_mean.unsqueeze(-2).expand(B, H, L_Q, V_mean.size(-1)).clone()
        else:
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            context = values.cumsum(dim=-2)
        return context

    # 更新上下文并计算出注意力权重，根据是否使用掩码来选择性地填充注意力得分。
    def _update_context(self, context, values, scores, index, L_Q, attention_mask):
        B, H, L_V, D = values.shape

        if self.mask_flag:
            attention_mask = prob_mask(B, H, L_Q, index, scores, device=values.device)
            scores.masked_fill_(attention_mask, -np.inf)

        attention = torch.softmax(scores, dim=-1)

        context[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = (
            attention @ values
        ).type_as(context)
        if self.output_attention:
            attentions = (torch.ones(B, H, L_V, L_V) / L_V).type_as(attention)
            attentions[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attention
            return context, attentions
        return context, None


# 实现了一个多头注意力机制，并对输入的查询、键和值进行了线性投影
class AttentionLayer(nn.Module):
    def __init__(self, attention, embed_dim, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (embed_dim // n_heads)  # d_keys 和 d_values 用于确定输出维度
        d_values = d_values or (embed_dim // n_heads)

        self.inner_attention = attention  # ProbSparseAttention or FullAttention
        self.query_attention = nn.Linear(embed_dim, d_keys * n_heads)
        self.key_projection = nn.Linear(embed_dim, d_keys * n_heads)
        self.value_projection = nn.Linear(embed_dim, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, embed_dim)  # 将多头注意力的输出维度变回embed_dim

        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attention_mask):
        B, L, _ = queries.shape  # B: batch_size 这里是B*N8, L: query序列长度197
        _, S, _ = keys.shape  # S: key序列长度
        H = self.n_heads  # 多头注意力头数

        queries = self.query_attention(queries).view(B, L, H, -1)  # B, L, H, d_keys
        keys = self.key_projection(keys).view(B, S, H, -1)  # B, S, H, d_keys
        values = self.value_projection(values).view(B, S, H, -1)  # B, S, H, d_values

        out, attention = self.inner_attention(queries, keys, values, attention_mask)   # out( B, L, H, d_values), attention( B, H, L, L)
        if self.mix:
            out = out.transpose(2, 1).contiguous()  # B, H, L, d_values
        out = out.view(B, L, -1)  # B, L, H * d_values

        return self.out_projection(out), attention  # B, L, d_model
