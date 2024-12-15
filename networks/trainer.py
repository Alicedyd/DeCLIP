import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.base_model import BaseModel
from models import get_model
from utils.utils import compute_batch_iou, compute_batch_localization_f1, compute_batch_ap

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt
        self.model = get_model(opt)
        
        # Initialize all possible parameters in the final layer
        for fc in self.model.fc:
            try:
                torch.nn.init.normal_(fc.weight.data, 0.0, opt.init_gain)
            except:
                pass

        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if "fc" in name and "resblock" not in name:
                    params.append(p) 
                else:
                    p.requires_grad = False
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.model.to(opt.gpu_ids[0])
        
        if opt.fully_supervised:
            self.ious = []
            self.F1_best = []
            self.F1_fixed = []
            self.ap = []
        elif opt.mask_plus_label:
            # xjw
            self.ious = []
            self.F1_best = []
            self.F1_fixed = []
            self.ap = []
            self.logits = []
            self.labels = []
        else:
            self.logits = []
            self.labels = []

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()
        
        # xjw
        if self.opt.mask_plus_label:
            self.mask = input[2].to(self.device).float()

    def forward(self):
        # self.output = self.model(self.input)
        self.output, self.output_2 = self.model(self.input, dual_output=self.opt.mask_plus_label) # output_2 is None when mask_plus_label=False, output is mask and output_2 is logit when mask_plus_label=True
        
        if self.opt.fully_supervised:
            # resize prediction to ground truth mask size
            if self.label.size()[1] != 256 * 256:
                label_size = (int(self.label.size()[1] ** 0.5), int(self.label.size()[1] ** 0.5))
                self.output = self.output.view(-1, 1, 256, 256)
                self.output = F.interpolate(self.output, size=label_size, mode='bilinear', align_corners=False)
                self.output = torch.flatten(self.output, start_dim=1).unsqueeze(1)
        
        # xjw
        if self.opt.mask_plus_label:
            if self.mask.size()[1] != 256*256:
                mask_size = (int(self.mask.size()[1] ** 0.5), int(self.mask.size()[1] ** 0.5))
                self.output = self.output.view(-1, 1, 256, 256)
                self.output = F.interpolate(self.output, size=label_size, mode='bilinear', align_corners=False)
                self.output = torch.flatten(self.output, start_dim=1).unsqueeze(1)

        if not self.opt.fully_supervised and not self.opt.mask_plus_label:
            self.output = torch.mean(self.output, dim=1)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        # outputs = self.output
        # xjw
        if not self.opt.mask_plus_label:
            outputs = self.output
        elif:
            mask = self.output
            logit = self.output_2
        
        if self.opt.fully_supervised:
            sigmoid_outputs = torch.sigmoid(outputs)
            
            # unflatten outputs and ground truth masks
            sigmoid_outputs = sigmoid_outputs.view(sigmoid_outputs.size(0), int(sigmoid_outputs.size(1)**0.5), int(sigmoid_outputs.size(1)**0.5))
            labels = self.label.view(self.label.size(0), int(self.label.size(1)**0.5), int(self.label.size(1)**0.5))

            iou = compute_batch_iou(sigmoid_outputs, labels)
            self.ious.extend(iou)

            F1_best, F1_fixed = compute_batch_localization_f1(sigmoid_outputs, labels)
            self.F1_best.extend(F1_best)
            self.F1_fixed.extend(F1_fixed)
            
            ap = compute_batch_ap(sigmoid_outputs, labels)
            self.ap.extend(ap)
        elif self.opt.mask_plus_label:
            # xjw
            sigmoid_mask = torch.sigmoid(mask)
            
            # unflatten mask and ground truth masks
            sigmoid_mask = sigmoid_mask.view(sigmoid_mask.size(0), int(sigmoid_mask.size(1)**0.5), int(sigmoid_mask.size(1)**0.5))
            mask = self.mask = self.mask.view(self.mask.size(0), int(self.mask.size(1)**0.5), int(self.mask.size(1)**0.5))
            
            iou = compute_batch_iou(sigmoid_mask, mask)
            self.ious.extend(iou)
            
            F1_best, F1_fixed = compute_batch_localization_f1(sigmoid_mask, mask)
            self.F1_best.extend(F1_best)
            self.F1_fixed.extend(F1_fixed)
            
            ap = compute_batch_ap(sigmoid_mask, mask)
            self.ap.extend(ap)
            
            self.logit.append(logit)
            self.labels.append(self.label)
            
        else:
            self.logits.append(outputs)
            self.labels.append(self.label)

        self.optimizer.zero_grad()
        
        # self.loss = self.loss_fn(outputs, self.label) 
        
        # xjw
        if self.opt.mask_plus_label:
            self.loss = self.loss_fn(mask, self.mask) + self.loss_fn(logit, self.label)
        else:
            self.loss = self.loss_fn(outputs, self.label)
        
        self.loss.backward()
        self.optimizer.step()

    def format_output(self):
        if not self.opt.fully_supervised:
            self.logits = torch.cat(self.logits, dim=0)
            self.labels = torch.cat(self.labels, dim=0)
