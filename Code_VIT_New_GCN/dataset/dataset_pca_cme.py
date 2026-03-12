import argparse
import os
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

from dataset.transforms import data_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class StackedDataset(Dataset):
    def __init__(self, img_path, new_data_path, para_path, test_fold=1, args=None, transform=None, mode="train"):
        super(StackedDataset, self).__init__()
        self.img_path = img_path
        self.para_path = para_path
        self.args = args
        self.transform = transform
        self.mode = mode  # "train" or "test"
        self.resize_img = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
        self.new_data_path = new_data_path  # 新的图像数据路径

        # 读取事件数据，包含新的图像路径
        self.event_data = self.read_event_data(img_path, para_path, test_fold, mode, new_data_path)

    def __len__(self):
        return len(self.event_data)

    def __getitem__(self, idx):
        # 获取当前事件的数据
        event = self.event_data[idx]

        # 加载图像数据
        img_list = []
        time_list = []
        new_img_list = []  # 新增图像列表
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像从 [0, 255] 范围缩放到 [0, 1]
            transforms.Normalize(mean=[0.529, 0.229, 0.077], std=[0.216, 0.076, 0.037])  # 标准化
        ])

        # 合并原始图像和新图像的路径
        all_img_paths = event['img_paths'] + event['new_img_paths']

        # 处理所有图像，应用相同的数据增强
        for img_path in all_img_paths:
            img_info = Image.open(img_path)
            img_info = img_info.convert("RGB")
            img_info = self.resize_img(img_info)  # 调用 resize_img 函数调整大小

            # 转换为tensor 或者 数据增强
            img_info = data_transforms(img_info, self.args)

            # 如果是原始图像，添加到 img_list
            if img_path in event['img_paths']:  # 原始图像
                img_list.append(img_info)
            else:  # 新图像
                new_img_list.append(img_info)

        # 处理原始图像的时间信息
        for time_str  in event['time_info']:
            # 获取时间信息
            time_list.append(time_str)

        # 在堆叠图像之前检查 img_list 是否为空
        if len(img_list) == 0 and len(new_img_list) == 0:
            print(f"Warning: No images loaded for index {idx}. Skipping this batch.")
            return None  # 返回一个合适的值，或者可以返回一个默认图像，避免继续执行 stack

        # 分别堆叠原始图像和新图像数据
        stacked_imgs = torch.stack(img_list)  # (N_original, C, H, W)
        stacked_imgs_new = torch.stack(new_img_list)  # (N_new, C, H, W)

        # 时间信息
        time_infor = torch.tensor([[int(ts.replace('-', '').replace(':', '').replace(' ', ''))] for ts in time_list],
                                  dtype=torch.long)   # 将时间信息转换为整数 去掉了符号和空格 (2,1)
        # 转换为 (2, 6) 格式
        output = []
        # 遍历每个时间字符串进行处理
        for i in range(time_infor.shape[0]):
            # 提取年份、月份、日期、小时、分钟、秒
            time_str = str(time_infor[i, 0].item())  # 获取每个时间字符串

            year = int(time_str[:4])  # 提取年份 (前4位)
            month = int(time_str[4:6])  # 提取月份 (第5-6位)
            day = int(time_str[6:8])  # 提取日期 (第7-8位)
            hour = int(time_str[8:10])  # 提取小时 (第9-10位)
            minute = int(time_str[10:12])  # 提取分钟 (第11-12位)
            second = int(time_str[12:14])  # 提取秒 (第13-14位)
            # 将提取的部分按顺序添加到列表中
            output.append([year, month, day, hour, minute, second])

        # 将输出转换为张量，形状为 (2, 6)
        time_info = torch.tensor(output, dtype=torch.long)

        # 加载物理参数和标签
        para_info = torch.tensor(event['para_info'], dtype=torch.float32)  # (12,)
        label = torch.tensor(event['label'], dtype=torch.float32)  # (1,)

        if stacked_imgs is None or stacked_imgs_new is None or time_info is None or para_info is None or label is None:
            # 如果有 None，返回一个默认值或跳过
            return None  # 或者返回默认数据

        # 返回堆叠后的原始图像、新图像、时间信息、物理参数、标签
        return stacked_imgs, stacked_imgs_new, time_info, para_info, label

    @staticmethod
    def read_event_data(data_path, para_path, test_fold, mode, new_data_path):
        """
        读取所有事件的数据包括：
        - img_paths: 原始图像路径
        - time_info: 原始时间信息
        - para_info: 物理参数
        - label: 标签
        - new_img_paths: 新图像路径 (从同级new_data目录读取)
        """
        event_data = []

        # 读取折信息，例如原始数据路径中的f1/f2/f3/f4
        folds = sorted(os.listdir(data_path))
        train_folds = [f for f in folds if f != f'f{test_fold}'] if mode == "train" else [f'f{test_fold}']

        for fold in train_folds:
            # 处理原始数据路径：data_path/fold/subfolder/...
            original_fold_path = os.path.join(data_path, fold)
            subfolders = sorted(os.listdir(original_fold_path))

            # 同时读取新数据路径：需要进入同名 fold 目录
            new_fold_path = os.path.join(new_data_path, fold)  # 这是关键的修正点！
            if not os.path.exists(new_fold_path):
                raise FileNotFoundError(f"新数据路径中缺少 {fold} 目录：{new_fold_path}")

            for subfolder in subfolders:
                # 原始数据子目录路径
                original_subfolder_path = os.path.join(original_fold_path, subfolder)
                group_id = subfolder.split('_')[1]  # 假设subfolder名称类似 "group_001"

                # 新数据子目录路径：必须与原始数据结构一致
                new_subfolder_path = os.path.join(new_fold_path, subfolder)
                if not os.path.exists(new_subfolder_path):
                    continue  # 或抛出错误，根据实际情况决定

                # 读取参数和标签（与原逻辑相同）
                para_file = os.path.join(para_path, f"f{test_fold}", f"{mode}_para.xlsx")
                para_df = pd.read_excel(para_file)
                group_data = para_df[para_df['Group_Number'].str.strip() == f'Group_{group_id}'.strip()]
                label = group_data['Label'].values[0]
                para_info = group_data.iloc[0, 4:16].tolist()

                # 方法一  裁剪极端值
                # para_info_clipped = np.clip(para_info, -1000, 1000)  # 将极端值裁剪到[-1000, 1000]范围

                # 方法二 对数变换，避免对数负值或零的问题，先将数据调整为正数
                para_info = np.array(para_info)
                para_info_log = np.log(para_info - np.min(para_info) + 1)

                # print("对数变换后的数据：")
                # print(para_info_log)

                para_info_log = para_info_log.reshape(-1, 1)

                '''使用 MinMaxScaler 进行归一化'''
                # scaler_1 = MinMaxScaler()
                # para_info_scaled_1 = scaler_1.fit_transform(para_info_log)
                # print("对数变换后归一化的结果MinMaxScaler：")
                # print(para_info_scaled_1)

                scaler_2 = StandardScaler()
                para_info_scaled_2 = scaler_2.fit_transform(para_info_log)
                para_info_scaled_2 = para_info_scaled_2.reshape(-1)  # 转回一维数组
                # print("对数变换后归一化的结果StandardScaler：")
                # print(para_info_scaled_2)

                # 读取原始图像
                original_paths, original_times = [], []
                for img_name in sorted(os.listdir(original_subfolder_path)):
                    if img_name.endswith('.png'):
                        original_paths.append(os.path.join(original_subfolder_path, img_name))
                        original_times.append(img_name.replace('.png', ''))

                # 读取新数据图像
                new_paths = []
                for img_name in sorted(os.listdir(new_subfolder_path)):
                    if img_name.endswith('.png'):
                        new_paths.append(os.path.join(new_subfolder_path, img_name))

                # 填充数据
                event_data.append({
                    'img_paths': original_paths,
                    'time_info': original_times,
                    'para_info': para_info_scaled_2,
                    'label': label,
                    'new_img_paths': new_paths  # ✅ 新增的图像路径
                })

        return event_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建解析器 配置参数

    img = r'/home/lm/CME-Project/DATA/data_image'
    pca_img = r'/home/lm/CME-Project/DATA/pca_pic'
    para = r'/home/lm/CME-Project/DATA/data_para'
    arg_s = parser.parse_args()  # 解析参数 将参数构造的类 赋给arg_s

    dataset = StackedDataset(img_path=img,
                             new_data_path=pca_img,
                             para_path=para,
                             test_fold=2,
                             args=arg_s,
                             transform=transforms.ToTensor(),
                             mode="train")
    image, pca_image, time_str, para, label = dataset.__getitem__(20)
    print(image.shape)

    for i in range(200):
        image, pca_image, time_str, para, label = dataset.__getitem__(i)
        print(image.shape)  # torch.Size([N, 3, 224, 224])
        # 获取 N 的值
        N = image.shape[0]

        # # 判断 N 是否为 66
        # if N != 2:
        #     print(f"Error: Expected N to be 66, but got {N}.")
        # else:
        #     print("N is correct, proceeding.")

    image, pca_image, time_str, para, label = dataset.__getitem__(1)
    print(image.shape)  # torch.Size([2, 3, 224, 224])

    # time_str ：tensor([[20120614141207], [20120614142405]])
    # 这个dataset中return出来的时间信息是以int型的形式存储的 这样的估计不能满足informer的输入要求
    print(image, time_str, para, label)

    # 测试单个事件的数据
    stacked_imgs,stacked_imgs_new, time_info, para_info, label = dataset[0]
    print("Stacked Images Shape:", stacked_imgs.shape)  # (N, C, H, W)
    print("stacked_imgs_new:",stacked_imgs_new.shape)
    print("Time Info Shape:", time_info.shape)  # (N, 1)
    print("Para Info Shape:", para_info.shape)  # (12,)
    print("Label Shape:", label.shape)  # ()

    # 测试 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    for batch_idx, (imgs, times, paras, labels) in enumerate(dataloader):
        print("Batch Stacked Images Shape:", imgs.shape)  # (B, N, C, H, W)
        print("Batch Time Info Shape:", times.shape)  # (B, N, 1)
        print("Batch Para Info Shape:", paras.shape)  # (B, 12)
        print("Batch Label Shape:", labels.shape)  # (B, 1)
        break