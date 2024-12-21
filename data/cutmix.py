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
import math

def crop_img(img, lam):
    """
    :param img: 输入图像 (Tensor)，前景(C, H, W) 格式的 Tensor
    :param lam: 混合比例 lambda，表示前景（贴的）占背景（被贴的）的比例
    :return: 裁剪的前景 (Tensor)
    """
    # 获取图像的尺寸
    H, W = img.shape[1], img.shape[2]  # 图像的高和宽
    lam = torch.tensor(lam) if isinstance(lam, np.ndarray) else lam
    # 裁剪区域的比例
    cut_rat = torch.sqrt(lam)  # 裁剪区域的比例
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # 计算中心裁剪的起始位置
    if W == cut_w:
        start_x = 0
    else:
        start_x = (W - cut_w) // 2
        
    if H == cut_h:
        start_y = 0
    else:
        start_y = (H - cut_h) // 2
    
    # 裁剪图像
    cropped_img = img[:, start_y:start_y + cut_h, start_x:start_x + cut_w]
    cropped_area = cropped_img.shape[1] * cropped_img.shape[2]  # 高度 * 宽度
    
    return cropped_img

def cutmix_data(img1=None, img2=None, label1=0, label2=1, mask=None, transform=None):
    """
    :cutmix 
    :param real_image: 真实图像 (Tensor)
    :param fake_image: 假图像 (Tensor)
    :param mask: 混合 mask
    return: 返回混合图像 (Tensor)
    """
    if isinstance(img1, str):
        img1, is_success = read_image(img1)
        img1, label1 = apply_transform(img1, label1, transform, is_dire=False)
    
    if isinstance(img2, str):
        # print('img2:', img2)
        img2, is_success = read_image(img2)
        img2, label2 = apply_transform(img2, label2, transform, is_dire=False)
    
    cutmix_img = mask * img1 + (1 - mask) * img2
    cutmix_label = 0 if label1 == 0 and label2 == 0 else 1

    return cutmix_img, cutmix_label

def mixup_data(img1, img2, label1, label2, alpha=1):
    """Compute the mixup data for binary classification (e.g., real/fake images). Return mixed inputs, mixed target, and lambda"""
    
    # 从 Beta 分布中采样一个 lam 值
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    # mixup img
    mixed_img = lam * img1 + (1 - lam) * img2

    # labels(0和0还是0，1和0为1)
    print(label1)
    print(label2)
    mixed_label = 0 if label1.item() == 0 and label2.item() == 0 else 1

    return mixed_img, mixed_label

def generate_mask(img, label, mixing_label, lam):
    """
    :param H: 图像的高度
    :param W: 图像的宽度
    :param label: 后景标签
    :param mixing_label: 前景标签
    :param lam: 混合比例 lambda，表示前景的比例（为0）
    
    :return: mxing_mask 用于混合后景和前景(后景 * mixing_mask + 前景 * (1 - mixing_mask)), 
             label_mask 用于训练的标签mask
    """
    # 转换 lam 为 tensor 类型
    # lam = torch.tensor(lam) if isinstance(lam, np.ndarray) else lam
    
    H, W = img.shape[1], img.shape[2]  
    
    # 裁剪区域的比例
    # cut_rat = torch.sqrt(lam)  # 裁剪区域的比例
    cut_rat = math.sqrt(lam) 
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # 计算中心裁剪的起始位置
    if W == cut_w:
        start_x = 0
    else:
        start_x = (W - cut_w) // 2
        
    if H == cut_h:
        start_y = 0
    else:
        start_y = (H - cut_h) // 2
    
    # 生成掩码：首先初始化一个全1的 mask
    mixing_mask = torch.ones((H, W), dtype=torch.float32)
    label_mask = torch.full((H, W), label, dtype=torch.float32)

    # 生成一个全0的 mask
    mixing_mask_crop = torch.zeros((cut_h, cut_w), dtype=torch.float32)
    label_mask_crop = torch.full((cut_h, cut_w), mixing_label, dtype=torch.float32)
    
    # 确保裁剪图像能够完全放置在背景图像上
    if cut_h > H or cut_w > W:
        raise ValueError("裁剪图像的尺寸大于背景图像，无法粘贴。")

    # 随机选择放置位置
    max_x = W - cut_w
    max_y = H - cut_h
    rand_x = random.randint(0, max_x)
    rand_y = random.randint(0, max_y)

    # 将全0mask图像粘贴到全1mask图像的随机位置
    mixing_mask[rand_y:rand_y + cut_h, rand_x:rand_x + cut_w] = mixing_mask_crop
    label_mask[rand_y:rand_y + cut_h, rand_x:rand_x + cut_w] = label_mask_crop

    return mixing_mask, label_mask