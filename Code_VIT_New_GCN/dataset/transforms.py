from torchvision import transforms

def data_transforms(image, args):
    """
    应用数据增强操作，仅包括旋转、平移和缩放操作。
    """
    if args is None:
        raise ValueError("参数 'args' 没有正确初始化或传递。")

    # 将字符串形式的元组转换为真实的元组
    random_crop_scale = tuple(map(float, args.random_crop_scale.strip('()').split(',')))
    random_crop_ratio = tuple(map(float, args.random_crop_ratio.strip('()').split(',')))
    translation_range = tuple(map(float, args.translation_range.strip('()').split(',')))

    # 数据增强操作的参数
    aug_args = {
        'random_crop': {'scale': random_crop_scale, 'ratio': random_crop_ratio, 'prob': args.random_crop_prob},
        'rotation': {'degrees': args.rotation_degrees, 'prob': args.rotation_prob,
                     'value_fill': args.rotation_value_fill},
        'translation': {'range': translation_range, 'prob': args.translation_prob,
                        'value_fill': args.translation_value_fill},
        # 添加 color_distortion 数据增强
            'color_distortion': {
                'brightness': args.color_distortion_brightness,
                'contrast': args.color_distortion_contrast,
                'saturation': args.color_distortion_saturation,
                'hue': args.color_distortion_hue,
                'prob': args.color_distortion_prob}
    }

    # 定义数据增强操作，仅包括旋转、平移和缩放
    operations = {
        'random_crop': random_apply(
            transforms.RandomResizedCrop(
                size=(224, 224),
                scale=aug_args['random_crop']['scale'],
                ratio=aug_args['random_crop']['ratio']),
                p=aug_args['random_crop']['prob']
                ),
        'rotation': random_apply(
            transforms.RandomRotation(
                degrees=aug_args['rotation']['degrees'],
                fill=aug_args['rotation']['value_fill']),
                p=aug_args['rotation']['prob']
                ),
        'translation': random_apply(
            transforms.RandomAffine(
                degrees=0,
                translate=aug_args['translation']['range'],
                fill=aug_args['translation']['value_fill']),
                p=aug_args['translation']['prob']
                ),
        'color_distortion': random_apply(
        transforms.ColorJitter(
            brightness=aug_args['color_distortion']['brightness'],
            contrast=aug_args['color_distortion']['contrast'],
            saturation=aug_args['color_distortion']['saturation'],
            hue=aug_args['color_distortion']['hue']),
            p=aug_args['color_distortion']['prob']
            ),
        
    }

    # 将启用的增强操作放入一个列表
    augmentations = []
    for op, op_func in operations.items():
        augmentations.append(op_func)

    # 标准化操作
    normalization = [
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.529, 0.229, 0.077], std=[0.216, 0.076, 0.037])
    ]

    # 创建完整的预处理流水线
    # 先进行数据增强 再tensor 标准化
    preprocess = transforms.Compose([
        *augmentations,
        *normalization
    ])

    # 应用预处理流水线
    return preprocess(image)


def random_apply(op, p):
    """应用随机选择操作的辅助函数"""
    return transforms.RandomApply([op], p=p)

