import torch
import torch.nn.functional as F
import math
from torch import nn
from torch.nn import Parameter


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):  # 初始化权重和偏置
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


''' 标准GCN '''
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, features, adj):
        x = F.relu(self.gc1(features, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)    
        # return F.log_softmax(x, dim=1)
        return F.relu(x)


# 计算余弦相似度得分矩阵 Xsc
def cosine_similarity_matrix(features):
    norm_features = F.normalize(features, p=2, dim=1)
    # norm_features = features
    Xsc = torch.matmul(norm_features, norm_features.T)
    return Xsc


# 构建邻接矩阵 adj
def build_adj_matrix(Xsc):
    """
    根据余弦相似度矩阵 Xsc 构建邻接矩阵 adj, 使用 A = Xsc @ Xsc^T 计算。
    """
    # 计算邻接矩阵 A = Xsc @ Xsc^T
    A = torch.matmul(Xsc, Xsc.T)

    # 可以设置一个阈值来确定哪些连接被保留
    adj = (A > 0).float()   # 假设阈值为0.5，可以调整
    # adj = (A > 0.5).to(dtype=torch.float)
    # adj = torch.tensor(A > 0.5, dtype=torch.float)

    # 清除自连接 (对角线元素设为0)
    adj.fill_diagonal_(0)

    return adj


# 完整的GCN网络
class GCN_with_cosine(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN_with_cosine, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, features):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        X_sc = cosine_similarity_matrix(features)
        X_sc = X_sc.to(device)
        adj = build_adj_matrix(Xsc=X_sc)

        x = F.relu(self.gc1(features, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return F.log_softmax(x, dim=1)
        return F.relu(x)
