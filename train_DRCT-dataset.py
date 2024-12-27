from copy import deepcopy
import os
import time
from tensorboardX import SummaryWriter

from validate import validate, validate_fully_supervised, validate_masked_detection, validate_masked_detection_v2
from utils.utils import compute_accuracy_detection, compute_average_precision_detection
# from data import create_dataloader
from data_DRCT.dataset_DRCT import AIGCDetectionDataset
from torch.utils.data import DataLoader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options_DRCT_dataset import TrainOptions
from utils.utils import derive_datapaths
import torch.multiprocessing
from data_DRCT.transform import create_train_transforms, create_val_transforms

from tqdm import tqdm
<<<<<<< HEAD
=======
from utils.visualize import *
>>>>>>> 237d3af5 (修改为DRCT数据集)

def get_val_opt(opt):
    val_opt = deepcopy(opt)
    val_opt.data_label = 'valid'
    return val_opt


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

<<<<<<< HEAD
    opt = TrainOptions().parse()
=======
    opt = TrainOptions().parse(print_options=False)
>>>>>>> 237d3af5 (修改为DRCT数据集)

    val_opt = get_val_opt(opt)
    is_one_hot = False
    
    model = Trainer(opt)
    
    xdl = AIGCDetectionDataset(opt.root_path, fake_root_path=opt.fake_root_path, fake_indexes=opt.fake_indexes, phase='train',
                               is_one_hot=is_one_hot, num_classes=opt.num_classes, inpainting_dir=opt.inpainting_dir, is_dire=opt.is_dire,
                               transform=create_train_transforms(size=opt.input_size, is_crop=opt.is_crop), 
                               prob_aug=opt.prob_aug, prob_cutmix=opt.prob_cutmix, prob_cutmixup_real_fake=opt.prob_cutmixup_real_fake, 
                               prob_cutmixup_real_rec=opt.prob_cutmixup_real_rec, prob_cutmixup_real_real=opt.prob_cutmixup_real_real
                               )
    sampler = None
    
    train_loader = DataLoader(xdl, batch_size=opt.batch_size, shuffle=sampler is None, num_workers=opt.num_threads, sampler=sampler)

    xdl_eval = AIGCDetectionDataset(opt.root_path, fake_root_path=opt.fake_root_path, fake_indexes=opt.fake_indexes, phase='val',
                                    num_classes=opt.num_classes, inpainting_dir=opt.inpainting_dir, is_dire=opt.is_dire,
                                    transform=create_val_transforms(size=opt.input_size, is_crop=opt.is_crop)
                                    )
    eval_loader = DataLoader(xdl_eval, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_threads)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    best_iou = 0
    
<<<<<<< HEAD
    for epoch in range(opt.niter):
=======
    visualize_mask=True ######
    if visualize_mask:
        # preparation for visualizing masks
        os.makedirs(os.path.join('train_vis', 'DRCT'), exist_ok=True)
        mask_save_path = os.path.join('train_vis', 'DRCT')
        os.makedirs(mask_save_path, exist_ok=True)
        
    for epoch in range(opt.niter):
        torch.cuda.empty_cache()
>>>>>>> 237d3af5 (修改为DRCT数据集)
        print(f"Epoch {epoch}")
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch [{epoch}/{opt.niter}]", unit="batch", ncols=150) as pbar:
            for i, data in enumerate(train_loader):
<<<<<<< HEAD
=======
                image, label, gd_masks, img_paths, cutmix_img_be_aug = data
                
>>>>>>> 237d3af5 (修改为DRCT数据集)
                model.total_steps += 1

                model.set_input(data)
                model.optimize_parameters()
                
                loss_value = round(model.loss.item(), 4)
                pbar.set_postfix(loss=loss_value)

                if model.total_steps % opt.loss_freq == 0:
                    # print(f"Train loss: {round(model.loss.item(), 4)} at step: {model.total_steps}; Iter time: {round((time.time() - start_time) / model.total_steps, 4)}")
                    epoch_loss += model.loss
                    train_writer.add_scalar('loss', model.loss, model.total_steps)
                    
                pbar.update(1)
<<<<<<< HEAD
=======
                
                ####### 可视化
                in_tens = image.cuda()
                outputs = model.model(in_tens)
                masks = outputs["mask"]
                
                pred_masks = []
                for i, mask in enumerate(masks):
                    if mask.size() != gd_masks[i].size():     
                        print('1111111111111111')
                        mask_resized = F.resize(mask.unsqueeze(0), gd_masks[i].size(), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).squeeze(0)
                        pred_masks.append(mask_resized)
                    else:
                        pred_masks.append(mask)
                        
                if visualize_mask and (model.total_steps % 100 == 0):
                    # visualize the masks
                    for i, (mask, gd_mask) in enumerate(zip(pred_masks, gd_masks)):
                        img_name = os.path.basename(img_paths[i])
                        img_prefix = img_name.split(".")[0]
                        if 'MSCOCO' in img_paths[i]:
                            file_name = f"{img_prefix}_0"
                        else:
                            file_name = f"{img_prefix}_1"
    
                        binary_mask = (mask > 0.5).float().cpu()
        
                        binary_mask = binary_mask.view(256, 256)##########
            
                        binary_mask = binary_mask.unsqueeze(0).unsqueeze(0)
                        binary_mask = torch.nn.functional.interpolate(binary_mask, size=(224, 224), mode='bilinear', align_corners=False)
                        binary_mask = binary_mask.squeeze(0).squeeze(0)
                    
        
                        # 应用预测掩码进行高低光融合
                        fused_img = apply_masked_highlight(cutmix_img_be_aug[i], binary_mask)
                        gd_mask = gd_mask.view(256, 256) ##########
                        # 可视化
                        visualize_fused_image(
                            img=cutmix_img_be_aug[i], 
                            gd_mask=gd_mask, 
                            pred_mask=binary_mask, 
                            fused_img=fused_img,
                            save_path=mask_save_path,
                            file_name=file_name,
                        )

>>>>>>> 237d3af5 (修改为DRCT数据集)

        epoch_loss /= len(train_loader)
        if opt.fully_supervised:
            # compute train performance  
            mean_iou = sum(model.ious)/len(model.ious)
            model.ious = []
            print(f"Epoch mean train IOU: {round(mean_iou, 2)}")
            
            mean_F1_best = sum(model.F1_best)/len(model.F1_best)
            model.F1_best = []
            print(f"Epoch mean train F1_best: {round(mean_F1_best, 4)}")
            mean_F1_fixed = sum(model.F1_fixed)/len(model.F1_fixed)
            model.F1_fixed = []
            print(f"Epoch mean train F1_fixed: {round(mean_F1_fixed, 4)}")
            
            mean_ap = sum(model.ap)/len(model.ap)
            model.ap = []
            print(f"Epoch mean train Mean AP: {round(mean_ap, 4)}")
        elif opt.mask_plus_label:
            # xjw
            mean_iou = sum(model.ious)/len(model.ious)
<<<<<<< HEAD
=======
            train_writer.add_scalar('iou', mean_iou, model.total_steps)
>>>>>>> 237d3af5 (修改为DRCT数据集)
            model.ious = []
            print(f"Epoch mean train IOU: {round(mean_iou, 2)}")
            
            mean_F1_best = sum(model.F1_best)/len(model.F1_best)
<<<<<<< HEAD
            model.F1_best = []
            print(f"Epoch mean train F1_best: {round(mean_F1_best, 4)}")
            mean_F1_fixed = sum(model.F1_fixed)/len(model.F1_fixed)
=======
            train_writer.add_scalar('F1_best', mean_F1_best, model.total_steps)
            model.F1_best = []
            print(f"Epoch mean train F1_best: {round(mean_F1_best, 4)}")
            
            mean_F1_fixed = sum(model.F1_fixed)/len(model.F1_fixed)
            train_writer.add_scalar('F1_fixed', mean_F1_fixed, model.total_steps)
>>>>>>> 237d3af5 (修改为DRCT数据集)
            model.F1_fixed = []
            print(f"Epoch mean train F1_fixed: {round(mean_F1_fixed, 4)}")
            
            model.format_output()
            mean_acc = compute_accuracy_detection(model.logits, model.labels)
<<<<<<< HEAD
            print(f"Epoch mean train ACC: {round(mean_acc, 2)}")
            mean_ap = compute_average_precision_detection(model.logits, model.labels)
=======
            train_writer.add_scalar('acc', mean_acc, model.total_steps)
            print(f"Epoch mean train ACC: {round(mean_acc, 2)}")
            
            mean_ap = compute_average_precision_detection(model.logits, model.labels)
            train_writer.add_scalar('ap', mean_ap, model.total_steps)
>>>>>>> 237d3af5 (修改为DRCT数据集)
            print(f"Epoch mean train AP: {round(mean_ap, 4)}")
        
            model.logits = []
            model.labels = []
        else:
            model.format_output()
            mean_acc = compute_accuracy_detection(model.logits, model.labels)
            print(f"Epoch mean train ACC: {round(mean_acc, 2)}")
            mean_ap = compute_average_precision_detection(model.logits, model.labels)
            print(f"Epoch mean train AP: {round(mean_ap, 4)}")
        
            model.logits = []
            model.labels = []

        # Validation
        model.eval()
        print('Validation')
        if opt.fully_supervised:
            ious, f1_best, f1_fixed, mean_ap, _ = validate_fully_supervised(model.model, eval_loader, opt.train_dataset)
            mean_iou = sum(ious)/len(ious)
            val_writer.add_scalar('iou', mean_iou, model.total_steps)
            print(f"(Val @ epoch {epoch}) IOU: {round(mean_iou, 2)}")
            
            mean_f1_best = sum(f1_best)/len(f1_best)
            val_writer.add_scalar('F1_best', mean_f1_best, model.total_steps)
            print(f"(Val @ epoch {epoch}) F1 best: {round(mean_f1_best, 4)}")
            
            mean_f1_fixed = sum(f1_fixed)/len(f1_fixed)
            val_writer.add_scalar('F1_fixed', mean_f1_fixed, model.total_steps)
            print(f"(Val @ epoch {epoch}) F1 fixed: {round(mean_f1_fixed, 4)}")
            
            mean_ap = sum(mean_ap)/len(mean_ap)
            val_writer.add_scalar('Mean AP', mean_ap, model.total_steps)
            print(f"(Val @ epoch {epoch}) Mean AP: {round(mean_ap, 4)}")
            
            # save best model weights or those at save_epoch_freq 
            if mean_iou > best_iou:
                print('saving best model at the end of epoch %d' % (epoch))
                model.save_networks( 'model_epoch_best.pth' )
                best_iou = mean_iou
            
            early_stopping(mean_iou, model)
        elif opt.mask_plus_label:
            ious, f1_best, f1_fixed, ap, acc, acc_best_thres, best_thres, _ = validate_masked_detection_v2(model.model, eval_loader)
            mean_iou = sum(ious)/len(ious)
            val_writer.add_scalar('iou', mean_iou, model.total_steps)
            print(f"(Val @ epoch {epoch}) IOU: {round(mean_iou, 2)}")
            
            mean_f1_best = sum(f1_best)/len(f1_best)
            val_writer.add_scalar('F1_best', mean_f1_best, model.total_steps)
            print(f"(Val @ epoch {epoch}) F1 best: {round(mean_f1_best, 4)}")
            
            mean_f1_fixed = sum(f1_fixed)/len(f1_fixed)
            val_writer.add_scalar('F1_fixed', mean_f1_fixed, model.total_steps)
            print(f"(Val @ epoch {epoch}) F1 fixed: {round(mean_f1_fixed, 4)}")
            
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            print(f"(Val @ epoch {epoch}) ACC: {acc}; ACC_BEST_THRES: {acc_best_thres}; AP: {ap}")
            
            # save best model weights or those at save_epoch_freq 
            if acc > best_iou:
                print('saving best model at the end of epoch %d' % (epoch))

                model.save_networks( 'model_epoch_best.pth' )
                best_iou = acc

            early_stopping(acc, model)
        else:
            ap, r_acc, f_acc, acc, _ = validate(model.model, eval_loader)
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            print(f"(Val @ epoch {epoch}) ACC: {acc}; AP: {ap}")

            # save best model weights or those at save_epoch_freq 
            if ap > best_iou:
                print('saving best model at the end of epoch %d' % (epoch))
                print(ap, best_iou)
<<<<<<< HEAD
                model.save_networks( 'model_epoch_best.pth' )
                best_iou = ap

=======
                model.save_networks(f'model_best_epoch_{epoch}_acc_{acc}.pth' )
                best_iou = ap
            model.save_networks(f'model_last_epoch_{epoch}_acc_{acc}.pth' )
>>>>>>> 237d3af5 (修改为DRCT数据集)
            early_stopping(acc, model)
        
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()
        print()