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

def get_val_opt(opt):
    val_opt = deepcopy(opt)
    val_opt.data_label = 'valid'
    return val_opt


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    opt = TrainOptions().parse()

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
    
    for epoch in range(opt.niter):
        print(f"Epoch {epoch}")
        epoch_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch [{epoch}/{opt.niter}]", unit="batch", ncols=150) as pbar:
            for i, data in enumerate(train_loader):
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
            model.ious = []
            print(f"Epoch mean train IOU: {round(mean_iou, 2)}")
            
            mean_F1_best = sum(model.F1_best)/len(model.F1_best)
            model.F1_best = []
            print(f"Epoch mean train F1_best: {round(mean_F1_best, 4)}")
            mean_F1_fixed = sum(model.F1_fixed)/len(model.F1_fixed)
            model.F1_fixed = []
            print(f"Epoch mean train F1_fixed: {round(mean_F1_fixed, 4)}")
            
            model.format_output()
            mean_acc = compute_accuracy_detection(model.logits, model.labels)
            print(f"Epoch mean train ACC: {round(mean_acc, 2)}")
            mean_ap = compute_average_precision_detection(model.logits, model.labels)
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
                model.save_networks( 'model_epoch_best.pth' )
                best_iou = ap

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