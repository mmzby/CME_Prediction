import os
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from typing import List
import numpy as np
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset

ImageFile.LOAD_TRUNCATED_IMAGES = True


# 时间特征类定义（保持原样）
class TimeFeature:
    # ... (省略，使用您提供的 TimeFeature 相关定义代码)
    pass


# 时间特征方法
def time_features_from_frequency(frequency: str) -> List[TimeFeature]:
    # ... (省略，使用您提供的 time_features_from_frequency 方法)
    pass


def time_features(dates, time_encoding=True, frequency: str = "h"):
    # ... (省略，使用您提供的 time_features 方法)
    pass


class StackedDataset(Dataset):
    def __init__(self, img_path, para_path, test_fold=1, transform=None, mode="train", time_frequency="h"):
        super(StackedDataset, self).__init__()
        self.img_path = img_path
        self.para_path = para_path
        self.transform = transform
        self.mode = mode
        self.time_frequency = time_frequency
        self.resize_img = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST)

        # 读取图像信息
        self.event_data = self.read_event_data(img_path, para_path, test_fold, mode)

    def __len__(self):
        return len(self.event_data)

    def __getitem__(self, idx):
        # 获取当前事件的数据
        event = self.event_data[idx]

        # 加载图像数据
        img_list = []
        for img_path in event['img_paths']:
            img = self.resize_img(Image.open(img_path).convert("RGB"))
            if self.transform:
                img = self.transform(img)
            img_list.append(img)

        # 堆叠图像数据
        stacked_imgs = torch.stack(img_list)  # (N, C, H, W)

        # 加载时间信息并生成时间特征
        dates_df = pd.DataFrame({'date': pd.to_datetime(event['time_info'])})
        time_info = time_features(dates_df, time_encoding=True, frequency=self.time_frequency)  # (N, F)

        # 加载物理参数和标签
        para_info = torch.tensor(event['para_info'], dtype=torch.float32)  # (12,)
        label = torch.tensor(event['label'], dtype=torch.float32)  # (1,)

        return stacked_imgs, torch.tensor(time_info, dtype=torch.float32), para_info, label

    @staticmethod
    def read_event_data(data_path, para_path, test_fold, mode):
        """
        读取所有事件的数据，包括图像路径、时间信息、物理参数和标签
        """
        event_data = []

        # 获取折信息
        folds = sorted(os.listdir(data_path))
        train_folds = [f for f in folds if f != f'f{test_fold}'] if mode == "train" else [f'f{test_fold}']

        for fold in train_folds:
            fold_path = os.path.join(data_path, fold)
            subfolders = sorted(os.listdir(fold_path))

            for subfolder in subfolders:
                subfolder_path = os.path.join(fold_path, subfolder)
                group_id = subfolder.split('_')[1]

                para_file = os.path.join(para_path, f"f{test_fold}", f"{mode}_para.xlsx")
                para_df = pd.read_excel(para_file)

                group_data = para_df[para_df['Group_Number'].str.strip() == f'Group_{group_id}'.strip()]
                label = group_data['Label'].values[0]
                para_info = group_data.iloc[0, 4:16].tolist()

                img_paths = []
                time_info = []

                images = sorted(os.listdir(subfolder_path))
                for img_name in images:
                    if not img_name.endswith('.png'):
                        continue
                    img_paths.append(os.path.join(subfolder_path, img_name))
                    time_info.append(img_name.replace('.png', ''))

                event_data.append({
                    'img_paths': img_paths,
                    'time_info': time_info,
                    'para_info': para_info,
                    'label': label
                })

        return event_data


if __name__ == '__main__':
    img = r'/home/lm/CME-Project/DATA/data_image'
    para = r'/home/lm/CME-Project/DATA/data_para'

    dataset = StackedDataset(
        img_path=img,
        para_path=para,
        test_fold=1,
        transform=transforms.ToTensor(),
        mode="train",
        time_frequency="h"
    )

    # 测试单个事件的数据
    stacked_imgs, time_features, para_info, label = dataset[0]
    print("Stacked Images Shape:", stacked_imgs.shape)  # (N, C, H, W)
    print("Time Features Shape:", time_features.shape)  # (N, F)
    print("Para Info Shape:", para_info.shape)  # (12,)
    print("Label Shape:", label.shape)  # (1,)

    # 测试 DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    for batch_idx, (imgs, times, paras, labels) in enumerate(dataloader):
        print("Batch Stacked Images Shape:", imgs.shape)  # (B, N, C, H, W)
        print("Batch Time Features Shape:", times.shape)  # (B, N, F)
        print("Batch Para Info Shape:", paras.shape)  # (B, 12)
        print("Batch Label Shape:", labels.shape)  # (B, 1)
        break
