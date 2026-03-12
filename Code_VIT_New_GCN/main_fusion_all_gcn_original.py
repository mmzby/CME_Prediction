import argparse
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torch.backends import cudnn
from models_cme.net_concat_all_net import GCN_FusionModule
from dataset.dataset_pca_cme import StackedDataset
import os
import torch
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import utils


# 定义训练参数
def train(args, model, train_dataloader, test_dataloader, optimizer,
          criterion_loss, test_fold):
    # start training
    epochs = args.num_epochs
    best_mae = 20  # 初始化最佳误差为20  起始误差应该是设定的阈值
    '''
        可以在这个地方导入loss函数 在下面for循环中直接用也可以
    '''
    # 记录训练loss
    Loss_list = []
    loss_L1 = nn.L1Loss()
    '''跑深度学习网络重点在于此 训练args.num_epochs个周期'''
    '''训练周期'''
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_loss_mse = 0.0
        train_loss_l1 = 0.0
        lr = utils.adjust_learning_rate(args, optimizer, epoch)
        train_bar = tqdm(train_dataloader, desc=f"Train phase: ", colour="RED")

        '''读取训练集的数据'''
        for batch_idx, (image, pca_image, time_str, para, label) in enumerate(train_bar):
            # print(time_str)
            # print(time_str.shape())
            # 这个train_bar就相当于dataloader的迭代器，每次迭代返回一个batch的数据
            # dataloader里面的数据怎么来的？ 就是从dataset里面读取的，dataset里面的数据是怎么来的？？
            # (image, time_str, para, label) 完全是和train_dataset.__getitem__()函数return的东西对应
            # dataset的getitem函数return的东西是什么？ return img, time_info, para_info, label
            # print(para.size()， time_str.size(), label.size())  image.size() 是(B, N,C, H, W)
            # 你在这里print的结果会和dataset的getitem函数里面print的结果对应
            '''--------------- 处理物理参数信息 ---------------'''
            # para = para.unsqueeze(dim=1)   # 将物理参数信息扩展一个维度 (2, 12) ----> (2, 1, 12)
            # print(para.size())

            ''' --------------- 处理标签信息 ---------------'''
            label = label.unsqueeze(dim=1)
            # 其实在dataset里面是空，这里面没有扩展之前应该是(B, )的形式 扩展之后变成了(B, 1)的形式
            # print(para.size()，label.size())   # 观察前后维度的变化

            '''--------------- 处理时间信息 ---------------'''
            # 在这个地方将时间信息处理成模型可以接受的形式
            # 把time_features这个库的time_features(dates, time_encoding=True, frequency: str = "h")
            # time_info = time_features(time_str, time_encoding=True, frequency: str = "h")\
            # 但是估计time_str这个要处理成和time_features.py文件里面dates的格式一致
            # 后面导入传入informer的时间信息就是time_info这个

            '''--------------- 将数据放在GPU上 --------------'''
            device = torch.device('cuda')
            # 将数据转移到GPU上 在这一步将数据都放在GPU上了 在此之前都是在CPU上
            # time_str = time_str.to(device)
            img, pca_image, para, label = image.to(device), pca_image.to(device), para.to(device), label.to(device)
            # print(img.size(), para.size(), label.size())  # 观察数据维度变化)

            ''' --------------- 将数据输入到模型中 ---------------'''
            predict = model(img, pca_image, para)
            # predict = model(img)  # 预测值  # img是(B, N,C, H, W)  predict是(B, 1)
            # predict, _ = model(img, para)

            ''' --------------- 计算损失 ---------------'''
            # 计算loss 在此之前把用到的loss都进行实例化 loss都是一个类 也是有forward函数的
            loss_mse = criterion_loss(predict, label)
            # loss_l1 = loss_L1(predict, label)
            loss_l1 = 0.0
            loss = loss_mse + loss_l1
            
            ''' --------------- 梯度反向传播 ---------------'''
            loss.requires_grad_(True)
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            # train_bar.update()

            ''' --------------- 计算loss总值 ---------------'''
            train_loss += loss.item()
            train_loss_mse += loss_mse.item()
            # train_loss_l1 += loss_l1.item()
            # 在不同时刻设置进度条的名字
            train_bar.desc = f"train epoch [{epoch + 1}/{epochs}] loss= {loss:.6f} lr= {lr:.6f}"

        ''' --------------- 一个周期训练结束 打印损失信息 ---------------'''
        # 到此为止 一个epoch的训练结束
        print("train_loss ----->", train_loss / (len(train_dataloader) * args.batch_size))  # 计算每张图片的损失
        print("train_loss_mse ----->", train_loss_mse / (len(train_dataloader) * args.batch_size))
        print("train_loss_l1 ----->", train_loss_l1 / (len(train_dataloader) * args.batch_size))

        ''' --------------- 开始测试 ---------------'''
        # print('-------one epoch train ending-------')
        # 进入测试阶段 每个epoch都进行测试 测试为了什么？？？得到mae, rmse 指标
        mae, rmse, sd = val(test_dataloader=test_dataloader,
                            test_fold=test_fold,
                            model=model,
                            checkpoint_dir=args.save_model_path,
                            epoch=epoch
                            )

        # 保存模型 根据指标的好坏去保存模型
        if mae <= best_mae and mae < 20:     # 误差最大为20, 小于20的保存下来
            best_mae = mae

            checkpoint_dir_root = args.save_model_path  # 保存最佳模型
            if not os.path.exists(checkpoint_dir_root):
                os.makedirs(checkpoint_dir_root)
            checkpoint_dir = os.path.join(checkpoint_dir_root, str(test_fold))

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            checkpoint_latest = os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar')
            # 保存最佳模型 在这个函数里面保存模型
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_mae,
            }, best_mae, mae, epoch, rmse, sd, checkpoint_dir, filename=checkpoint_latest)

        elif mae > best_mae:
            best_mae = best_mae
        # 到此为止是什么？ 一个epoch结束 开始打印信息供monitor观察 然后开始下一个epoch的训练
        print('best_MAE:', best_mae, 'MAE:', mae, 'RMSE:', rmse, 'SD:', sd, '\n')

        # 每10个epoch保存一次损失值
        if epoch % 20 == 0:
            Loss_list.append(train_loss / (len(train_dataloader) * args.batch_size))

    ''' --------------- 损失函数的可视化---------------'''
    num_saved_points = len(Loss_list)

    if num_saved_points == 0:
        raise ValueError("Loss_list is empty. Nothing to plot.")

    # 自动计算保存间隔
    save_interval = epochs // num_saved_points  # 如 epochs=20 且1点 → 20

    # 构造横轴坐标
    x = list(range(save_interval, epochs + 1, save_interval))
    y = Loss_list

    # 容错处理：如果记录少于应有的点数，自动截断 x 保证长度一致
    x = x[:len(y)]

    plt.figure(figsize=(18, 8))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(x)
    plt.grid()
    plt.tight_layout()
    plt.savefig(checkpoint_dir, dpi=300)
    plt.show()


# 对训练好的模型进行验证，计算模型在测试数据集上的平均绝对误差（MAE）和均方根误差（RMSE）
def val(test_dataloader, test_fold, model, checkpoint_dir=None, epoch=0):
    print('Start Validation!')
    print("test_fold: 第{0}折".format(test_fold))
    total_mae, total_rmse, total_sd = 0, 0, 0
    # total_mae, total_rmse, total_me, total_sd, total_mre = 0, 0, 0, 0, 0

    with torch.no_grad():
        model.eval()
        test_bar = tqdm(test_dataloader, desc=f"Test phase: ", colour="RED")

        for batch_idx, (image, pca_image, time_str, para, label) in enumerate(test_bar):
            # para = para.unsqueeze(dim=1)
            label = label.unsqueeze(dim=1)
            # print(para.size())  # torch.Size([4, 1, 12])
            # print(label.size())  # torch.Size([4, 1])

            # 将数据放在 GPU 上
            device = torch.device('cuda')
            img, pca_image, para, label = image.to(device), pca_image.to(device), para.to(device), label.to(device)

            predict = model(img, pca_image, para)

            # 去掉 batch 维度,这样只剩下batch个数值，方便计算
            predict = predict.squeeze(dim=1)  # (batch,)
            label = label.squeeze(dim=1)  # (batch,)
            # print(predict.size())
            # print(label.size())

            # 计算误差并累积 （mean均值, 计算了当前batch中图片平均的MAE）
            total_mae += torch.mean(torch.abs(label - predict)).item()
            total_rmse += torch.sqrt(torch.mean((label - predict) ** 2)).item()
            # total_me += torch.mean(label - predict).item()
            total_sd += torch.sqrt(torch.sum((label - predict - torch.mean(label - predict)) ** 2) / ((label - predict).numel() - 1)).item()
            # total_mre += torch.mean(torch.abs(label - predict)/label).item()

    # 平均 MAE 和 RMSE
    mae = round(total_mae / len(test_dataloader), 3)        # 计算每张图片的误差, 保留三位小数
    rmse = round(total_rmse / len(test_dataloader), 3)
    # me = round(total_me / len(test_dataloader), 3)
    sd = round(total_sd / len(test_dataloader), 3)  
    # mre = round(total_mre / len(test_dataloader), 2)

    return mae, rmse, sd

def main(args, test_fold=1):
    print(args.dataset_image_path)  # 打印了cfg对象中的base属性中的save_path属性的值
    print(args.dataset_pca_img_path)

    '''加载保存路径save_path log_path: checkpoints'''
    train_dataset = StackedDataset(img_path=args.dataset_image_path,
                                   new_data_path=args.dataset_pca_img_path,
                                   para_path=args.dataset_para_path,
                                   test_fold=test_fold,
                                   args=args,
                                   transform=transforms.ToTensor(),
                                   mode="train")

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,  # True
                                  pin_memory=True,  # 是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
                                  num_workers=2,
                                  drop_last=True)

    test_dataset = StackedDataset(img_path=args.dataset_image_path,
                                  new_data_path=args.dataset_pca_img_path,
                                  para_path=args.dataset_para_path,
                                  test_fold=test_fold,
                                  args=args,
                                  transform=transforms.ToTensor(),
                                  mode="test")

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 pin_memory=True,  # 是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
                                 num_workers=2,
                                 drop_last=True)

    # 网络模型
    # .to(device) 网络模型也一定要放在GPU上（不仅是输入数据）
    model = GCN_FusionModule(para_dim=12, num_classes=1).to(device)

    # model = vit.vit_base_patch16_224(num_classes=args.num_classes).to(device)
    # 加载训练好的权重
    # checkpoint_path = "/home/lm/CME-Project/MODEL/vit_sparse5_model/3/model_031_12.3120_15.5750.pth.tar"
    # checkpoint = torch.load(checkpoint_path)  # , map_location=device
    # model.load_state_dict(checkpoint['state_dict'])
    # 将模型放到指定设备上 (例如：GPU 或 CPU)

    cudnn.benchmark = True  # 让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()  # 在模块级实现数据并行（多GPU）
    # 打印模型结构
    # print(model)

    # 优化器 有三种---具体哪种好，还要通过实验来验证--- 每种优化器的学利率不同
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 betas=args.betas,
                                 weight_decay=args.weight_decay)  # lr:0.0001 数量级  weight_decay 5e-4
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)   # lr:0.001 到 0.01 weight_decay 1e-4
    # loss 有两种（一般使用L2）
    loss_function = nn.MSELoss()  # 这个是标准的库函数

    # loss_function = nn.HuberLoss()
    # loss_function = nn.L1Loss()

    print("training beginning!")

    '''调用名为train的函数 并传递了多个参数来进行模型的训练。 
    这些参数包括配置信息、模型对象、训练数据集、测试数据集和估计器对象。'''
    train(args=args,
          model=model,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          optimizer=optimizer,
          criterion_loss=loss_function,    # 损失函数
          test_fold=test_fold)  # 训练模型
    print("training finished!")


if __name__ == '__main__':
    # 服务器GPU的地址
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda')

    parser = argparse.ArgumentParser()  # 创建解析器 配置参数

    '''basic configuration'''
    parser.add_argument('--num_classes', type=int, default=1)       # 做预测任务
    parser.add_argument('--num_epochs', type=int, default=500)      # 训练轮数
    parser.add_argument('--batch_size', type=int, default=4)        # batch_size
    parser.add_argument('--k_fold', type=int, default=6)
    parser.add_argument('--use_gpu', type=str, default=True)
    parser.add_argument('--criterion', type=str, default=True)

    '''optimization'''
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_mode', type=str, default='poly')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Adam β1 β2')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2正则化系数 1e-4')

    '''model dataset path'''
    parser.add_argument('--dataset_image_path', type=str, default="/data/flower_photos")
    parser.add_argument('--dataset_para_path', type=str, default="/data/flower_photos")
    parser.add_argument('--save_model_path', type=str, default="/data/flower_photos")
    # parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    '''model'''
    parser.add_argument('--init_feature_channels', type=int, default=512, help='initial feature channels for vit')
    parser.add_argument('--add_attention', type=bool, default=True, help='whether to add attention to model')

    '''数据增强参数'''
    parser.add_argument('--random_crop_scale', type=str, default="(0.8, 1.0)",
                        help='random crop scale range (min, max)')
    parser.add_argument('--random_crop_ratio', type=str, default="(1, 1)",
                        help='random crop aspect ratio range (min, max)')
    parser.add_argument('--random_crop_prob', type=float, default=0.5, help='probability of applying random crop')

    parser.add_argument('--horizontal_flip_prob', type=float, default=0.5, help='probability of horizontal flip')
    parser.add_argument('--vertical_flip_prob', type=float, default=0.5, help='probability of vertical flip')

    parser.add_argument('--color_distortion_brightness', type=float, default=0.2,
                        help='brightness for color distortion')
    parser.add_argument('--color_distortion_contrast', type=float, default=0.2, help='contrast for color distortion')
    parser.add_argument('--color_distortion_saturation', type=float, default=0.2,
                        help='saturation for color distortion')
    parser.add_argument('--color_distortion_hue', type=float, default=0.2, help='hue for color distortion')
    parser.add_argument('--color_distortion_prob', type=float, default=0.5,
                        help='probability of applying color distortion')

    parser.add_argument('--rotation_degrees', type=int, default=30, help='max degrees for rotation')
    parser.add_argument('--rotation_prob', type=float, default=0.5, help='probability of applying rotation')
    parser.add_argument('--rotation_value_fill', type=int, default=0, help='fill value for rotation')

    parser.add_argument('--translation_range', type=str, default="(0.1, 0.1)",
                        help='translation range for affine transformation')
    parser.add_argument('--translation_prob', type=float, default=0.5, help='probability of applying translation')
    parser.add_argument('--translation_value_fill', type=int, default=0, help='fill value for translation')

    parser.add_argument('--grayscale_prob', type=float, default=0.2, help='probability of applying grayscale')

    arg_s = parser.parse_args()  # 解析参数 将参数构造的类 赋给arg_s

    k_fold = arg_s.k_fold
    arg_s.save_model_path = r'/mnt/500T/homes/lm/CME-Project/MODEL/gcn'  # 模型结果保存路径

    arg_s.dataset_image_path = r'/home/lm/CME-Project/DATA/renew_data/data_image'
    arg_s.dataset_pca_img_path=r'/mnt/disk16T/lm/data/gray_data'
    arg_s.dataset_para_path = r'/home/lm/CME-Project/DATA/renew_data/data_para'

    for i in range(k_fold):
        main(args=arg_s, test_fold=int(i + 1))



