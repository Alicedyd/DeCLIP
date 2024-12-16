 python validate.py --arch=CLIP:ViT-L/14 --ckpt=/root/autodl-tmp/code/DeCLIP/checkpoint/experiment_name/model_epoch_best.pth \
                    --result_folder=/root/autodl-tmp/code/DeCLIP/results --gpu_ids 1 \
                    --mask_plus_label \
                    --batch_size 16 
