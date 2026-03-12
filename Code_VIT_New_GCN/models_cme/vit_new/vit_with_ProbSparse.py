"""
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import os
from functools import partial
from collections import OrderedDict
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_cme.vit_new.attention import AttentionLayer, FullAttention
from models_cme.vit_new.encoder import Encoder, EncoderLayer
# from attention import AttentionLayer, FullAttention
# from encoder import Encoder, EncoderLayer


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape

        # H == self.img_size[0]
        # W == self.img_size[1]

        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


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
        B, L_Q, H, D = queries.shape  # B: batch_size, L_Q: query序列长度, H: 多头注意力头数, D: 输入维度 （8B*N，197，12，64）
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
        K_expand = keys.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # （8，12，197，197，64）
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor * ln(L_K)) * L_Q   # （197，30）
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # （8，12，197，30，64）
        Q_K_sample = (queries.unsqueeze(-2) @ K_sample.transpose(-2, -1)).squeeze()  # （8，12，197，30）

        # find the top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # （8，12，197）
        M_top = M.topk(n_top, sorted=False)[1]  # （8，12，30）

        # use the reduced Q to calculate Q_K
        Q_reduce = queries[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]   # factor * ln(L_Q)，（8，12，30，64）
        Q_K = Q_reduce @ keys.transpose(-2, -1)  # factor * ln(L_Q) * L_K，（8，12，30，197）

        return Q_K, M_top

    # 根据是否使用掩码来计算初始上下文。对于无掩码的情况，使用 values 的平均值；若使用掩码，则 L_Q 必须等于 L_V
    def _get_initial_context(self, values, L_Q):
        B, H, L_V, D = values.shape  # （8，12，197，64）
        if not self.mask_flag:
            V_mean = values.mean(dim=-2)  # （8，12，64）
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



class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,dropout=0.05,
                 embed_dim=768, depth=12, attention_type="prob", output_attention=False,
                 factor=5, n_heads=8, activation="gelu", distil=True, d_ff=2048,
                 representation_size=None, distilled=False, drop_ratio=0., embed_layer=PatchEmbed):
        """
        Args:
            in_c (int): 输入通道数 num_classes (int): 分类任务的类别数
            embed_dim (int): 每个patch的嵌入维度 depth (int): transformer的层数
            num_heads (int): 每层中多头自注意力的头数
            mlp_ratio (int): MLP 层中隐藏层维度与嵌入维度的比值
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): 是否包含 DeiT 风格的蒸馏头
            embed_layer (nn.Module): 用于生成patch嵌入的层
            norm_layer: (nn.Module): normalization layer
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models_stack
        self.num_tokens = 2 if distilled else 1

        self.patch_embed = embed_layer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_c=in_c,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        Attention = ProbSparseAttention if attention_type == "prob" else FullAttention

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        embed_dim,
                        n_heads,  # 多头注意力头数
                        mix=False,
                    ),
                    embed_dim,  # 模型维度
                    d_ff,  # fcn 前馈网络的维度
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(depth)
            ],
            [SelfAttentionDistil(embed_dim) for _ in range(depth - 1)] if distil else None,
            nn.LayerNorm(embed_dim),
        )

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)  # [B*N , 196, 768] 分块嵌入
        # [1, 1, 768] -> [B*N , 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # 假设 cls_token 是 ViT 的位置编码，你可以通过线性层进行转换
        # cls_token = self.linear(cls_token)  # [B, 1, 768] -> [B, 1, 640]
        # cls_token_en = torch.cat((cls_token, time_feat), dim=2)
        # print(cls_token_en.shape)

        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B*N, 197, 768] 沿着第 1 维拼接
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        # print(x.shape)  # torch.Size([8, 197, 768])

        enc_out, attentions = self.encoder(x)
        # print(enc_out.shape)  # torch.Size([8, 3, 768])
        # print(len(attentions))  # 用来查看中间层特征 可将注意力矩阵可视化
        # print(attentions)

        enc_out = enc_out[:, -1, :]  
        # print(enc_out.shape)   # torch.Size([8, 768])

        if self.dist_token is None:
            return self.pre_logits(enc_out)

    def forward(self, x):
        # 解包输入的形状 [B, N, C, H, W] -> [B, N, num_patches, embed_dim]
        B, N, C, H, W = x.shape

        # 合并维度
        x = x.view(-1, C, H, W)
        # print(x.shape)
        x = self.forward_features(x)  # torch.Size([8, 768])
        vit_feat = x 
        # time_feat = time_feat.view(-1, F1)
        # time_feat = time_feat.unsqueeze(1)
        # x = self.forward_features(x, time_feat)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)    # torch.Size([8, 1])
            # print(x.shape)
            # 获取 x 的形状并解包出 B 和 N
            x = x.view(B, N, -1)
            pre = x.mean(dim=1)   # torch.Size([4, 1])

            vit_feat = vit_feat.view(B, N, -1)
            vit_feat = vit_feat.mean(dim=1)

        return pre, vit_feat


def prob_mask(B, H, L, index, scores, device):
    mask = torch.ones(L, scores.shape[-1], dtype=torch.bool, device=device).triu(1)
    mask_ex = mask[None, None, :].expand(B, H, L, scores.shape[-1])
    indicator = mask_ex[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
    mask = indicator.view(scores.shape)
    return mask


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              n_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              n_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224(num_classes: int = 1000):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              n_heads=12,
                              representation_size=None,
                              num_classes=num_classes)
    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(img_size=224,
                              patch_size=32,
                              embed_dim=768,
                              depth=12,
                              n_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    device = torch.device('cuda')

    # 输入到五维张量
    image = torch.randn(4, 2, 3, 224, 224)
    time_f = torch.randn(4, 2, 128)
    image = image.clone().detach().to(dtype=torch.float32)
    image = image.to(device)
    time_f = time_f.to(device)
    # 合并维度 (4, 2, 3, 224, 224) -> (8, 3, 224, 224)
    B, N, C, H, W = image.shape

    model = vit_base_patch32_224(num_classes=1).to(device)  # vit_base_patch32_224 基础版
    # feat, preds = model(image, time_f)
    preds, vit_feat = model(image)
    print(vit_feat.shape)
    print(preds.shape)
    # 将预测结果还原为原始 batch 格式 (4, 2, num_classes)
    # preds = preds.view(B, N, -1)
    # print(preds)

    # assert preds.shape == (1, 2), 'correct logits outputted'
