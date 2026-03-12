import os.path as osp
import shutil
import torch


# 保存模型，state:先建立一个字典，保存三个参数
def save_checkpoint(state, best_mae, mae, epoch, rmse, sd, checkpoint_path, filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if mae:
        # shutil.copyfile(src, dst):将名为src的文件的内容复制到名为dst的文件中,dst必须是完整的目标文件名
        shutil.copyfile(filename,
                        osp.join(checkpoint_path,
                                 'model_{:03d}_{:.4f}_{:.4f}_{:.4f}.pth.tar'.format((epoch + 1),
                                                                             best_mae, rmse, sd)))


def adjust_learning_rate(opt, optimizer, epoch):  # 设置学习率
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))  # **表示取幂运算，//表示向下取整除
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9  # 指数衰减
    elif opt.lr_mode == 'kbc':
        lr = opt.lr / (1 + 10 ** (-4) * opt.num_epochs)
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
