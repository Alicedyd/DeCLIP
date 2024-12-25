# cutmix or mixup 
import os
import csv
import numpy as np
from PIL import Image
import glob
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import itertools
import torch
import torchvision
import math
from pathlib import Path

def read_image(image_path, resize_size=None):
    # try:
    #     image = cv2.imread(image_path)
    #     if resize_size is not None:
    #         image = resize_long_size(image, long_size=resize_size)
    #     # Revert from BGR
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     return image, True
    # except:
    #     print(f'{image_path} read error!!!')
    #     return np.zeros(shape=(512, 512, 3), dtype=np.uint8), False
    
    try:
        # 使用Pillow加载图像，Pillow会自动处理ICC配置文件
        image = Image.open(image_path)
        
        # 如果需要调整大小
        if resize_size is not None:
            image = image.resize((resize_size, resize_size))
        
        # 转换为 NumPy 数组并确保 RGB 格式
        image = np.array(image.convert("RGB"))
        
        return image, True
    except Exception as e:
        print(f'{image_path} read error!!! {e}')
        return np.zeros(shape=(512, 512, 3), dtype=np.uint8), False

def generate_patch_mask(img, lam):
    """
    :param img: 输入图像，形状为 (C, H, W)
    :param lam: 混合比例 lambda，表示前景的比例（为0）
    :return: mask, 后景 * mask + 前景 * (1 - mask)
    """
    H, W = img.shape[1], img.shape[2]  # 获取图像的高度和宽度

    # 定义 patch 的大小
    patch_size = 14

    # 计算 patch 的数量
    patch_H_number = H // patch_size
    patch_W_number = W // patch_size

    # 初始化一个全1的 mask，与图像大小相同
    mask = torch.ones((H, W), dtype=torch.float32)

    # 计算要置为0的patch数量，基于 lambda
    num_patches = patch_H_number * patch_W_number
    num_zero_patches = int(num_patches * (1 - lam))

    # 随机选择若干个 patch 的索引，将其置为 0
    zero_indices = random.sample(range(num_patches), num_zero_patches)
    for idx in zero_indices:
        row = idx // patch_W_number
        col = idx % patch_W_number
        start_y = row * patch_size
        start_x = col * patch_size

        # 将对应的 14x14 区域置为 0
        mask[start_y:start_y + patch_size, start_x:start_x + patch_size] = 0

    return mask

def apply_transform(image1, image2, label1, label2, transform, is_dire=False):   
    # 只在 transform 存在并且 is_dire 为 False 时应用 transform
    if transform is not None and not is_dire:
        if image2 is None:
            try:
                if isinstance(transform, torchvision.transforms.transforms.Compose):
                    image1 = transform(Image.fromarray(image1))
                else:
                    data = transform(image=image1)
                    image1 = data["image"]
            except Exception as e:
                print(f"Transform error: {e}")
                print('-------------------------')
                # 在转换失败时，返回默认的零填充图像
                image1 = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
                if isinstance(transform, torchvision.transforms.transforms.Compose):
                    image1 = transform(Image.fromarray(image1))
                else:
                    data = transform(image=image1)
                    image1 = data["image"]
                label1 = 0
            return image1, label1
            
        else:
            try:
                if isinstance(transform, torchvision.transforms.transforms.Compose):
                    image1 = transform(Image.fromarray(image1))
                    image2 = transform(Image.fromarray(image2))
                else:
                    data = transform(image=image1, rec_image=image2)
                    image1 = data["image"]
                    image2 = data["rec_image"]
            except Exception as e:
                print(f"Transform error: {e}")
                print('-------------------------')
                # 在转换失败时，返回默认的零填充图像
                image1 = np.zeros(shape=(512, 512, 3), dtype=np.uint8)
                if isinstance(transform, torchvision.transforms.transforms.Compose):
                    image1 = transform(Image.fromarray(image1))
                else:
                    data = transform(image=image1)
                    image1 = data["image"]
                label1 = 0
                image2 = None
                label2 = 0     

            return image1, image2, label1, label2


def cutmix_data(img1_path=None, img2_path=None, label1=0, label2=1, mask=None, transform=None):
    """
    :cutmix 
    :param real/fake image imput
    :param mask: 混合 mask
    return: 返回混合图像 (Tensor)
    """
    img1, _ = read_image(img1_path)
    img2, _ = read_image(img2_path)
    
    if 'inpainting' in img2_path: # real和rec一起变换
        img1, img2, label1, label2 = apply_transform(img1, img2, label1, label2, transform, is_dire=False)
        
    else:
        img1, label1 = apply_transform(img1, None, label1, None, transform, is_dire=False)
        img2, label2 = apply_transform(img2, None, label2, None, transform, is_dire=False)
    
    img1_label = torch.full((img1.shape[1], img1.shape[2]), label1, dtype=torch.float32)
    img2_label = torch.full((img2.shape[1], img2.shape[2]), label2, dtype=torch.float32)
    mask_label = mask * img1_label + (1 - mask) * img2_label
    
    cutmix_img = mask * img1 + (1 - mask) * img2
    cutmix_label = 0 if label1 == 0 and label2 == 0 else 1

    return cutmix_img, cutmix_label, mask_label[0, :, :]

def mixup_data(img1_path=None, img2_path=None, mask=None, alpha=None, transform=None):
    """
    :mixup 
    :param real_image: 真实图像 (Tensor)
    :param fake_image: 假图像 (Tensor)
    :param mask: 混合 mask(需要mixup的区域)
    :alpha: mixup的比例
    return: 返回混合图像 (Tensor)
    """
    real_img, _ = read_image(img1_path)
    fake_img, _ = read_image(img2_path)
    real_img, fake_img, label1, label2 = apply_transform(real_img, fake_img, 0, 1, transform, is_dire=False)
        
    mixup_fake_real = alpha * real_img + (1 - alpha) * fake_img
    mixed_img = mask * real_img + (1 - mask) * mixup_fake_real    
    mixed_label = 1

    mask_label = torch.full((real_img.shape[1], real_img.shape[2]), 1, dtype=torch.float32)
    
    return mixed_img, mixed_label, mask_label

if __name__ == '__main__':
    import torch
    import math
    import random
    import torchvision.transforms as transforms
    from PIL import Image
    lam = 0.5
    # 随机生成一个 224x224 的图像
    img = torch.rand((3, 224, 224), dtype=torch.float32)
    mask = generate_patch_mask(img, lam)
    # 保存 mask 为图像文件
    mask_image = transforms.ToPILImage()(mask.unsqueeze(0))
    mask_image.save("mask.png")