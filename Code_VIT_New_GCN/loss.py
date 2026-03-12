import torch
import torch.nn as nn


# 暂时没用（以后自己设计损失函数的时候，可以放在这个文件里面）
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.35, ga_ma=2, gamma2=3):
        super(BCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = ga_ma
        self.gamma2 = gamma2

    def forward(self, logits, label):
        # label = label.unsqueeze(1)  #  size(N, 1)
        assert label.size() == logits.size()
        probs = torch.sigmoid(logits)
        pos_loss = (-label*self.alpha*probs.log()*(1-probs)**self.gamma)
        neg_loss = -(1-label)*(1-self.alpha)*(1-probs).log()*probs**self.gamma
        loss = (pos_loss + neg_loss).mean()
        return loss


class KappaLoss(nn.Module):
    def __init__(self, num_classes, y_pow=2, eps=1e-10):
        super(KappaLoss, self).__init__()
        self.num_classes = num_classes
        self.y_pow = y_pow
        self.eps = eps

    def kappa_loss(self, y_pred, y_true):
        num_classes = self.num_classes
        y = torch.eye(num_classes).cuda()
        y_true = y[y_true]

        y_true = y_true.float()
        repeat_op = torch.Tensor(list(range(num_classes))).unsqueeze(1).repeat((1, num_classes)).cuda()
        repeat_op_sq = torch.square((repeat_op - repeat_op.T))
        weights = repeat_op_sq / ((num_classes - 1) ** 2)

        pred_ = y_pred ** self.y_pow
        pred_norm = pred_ / (self.eps + torch.reshape(torch.sum(pred_, 1), [-1, 1]))

        hist_rater_a = torch.sum(pred_norm, 0)
        hist_rater_b = torch.sum(y_true, 0)

        conf_mat = torch.matmul(pred_norm.T, y_true)

        bsize = y_pred.size(0)
        nom = torch.sum(weights * conf_mat)
        expected_probs = torch.matmul(torch.reshape(hist_rater_a, [num_classes, 1]), torch.reshape(hist_rater_b, [1, num_classes]))
        denom = torch.sum(weights * expected_probs / bsize)

        return nom / (denom + self.eps)

    def forward(self, y_pred, y_true):
        return self.kappa_loss(y_pred, y_true)
        # 也是nn.Module这个标准类定义自己的loss类 里面包含def __init__和forward方法
        # 不论model还是loss都是nn.Module这个类封装的 我们运行的时候 它自动就去执行forward方法
        # 这是自己设计的loss函数


class arc_smooth_L1_Loss(nn.Module):
    def __init__(self, beta):
        super(arc_smooth_L1_Loss, self).__init__()
        self.beta = beta

    def forward(self, input, target):
        return arc_smooth_L1_loss(input, target, self.beta)


def arc_smooth_L1_loss(input, target, beta=2):
    loss_part1 = torch.abs(input - target)
    loss_part2 = loss_part1 ** 2

    loss2 = torch.where(loss_part1 <= beta, loss_part1, loss_part2)
    loss2 = torch.mean(loss2)

    return loss2