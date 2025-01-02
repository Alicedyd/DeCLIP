from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.ops import misc

import torch

from m_rcnn import maskrcnn

if __name__ == "__main__":
    # Dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)  # [B, 3, 224, 224]
    gt_masks = torch.randint(0, 2, (batch_size, 224, 224)).float()  # [B, 224, 224]
    gt_labels = torch.randint(0, 2, (batch_size,))  # [B]

    # Model
    model = maskrcnn.maskrcnn_resnet50(pretrained=False, num_classes=1)

    # Forward pass
    outputs = model(images)
    print(outputs)