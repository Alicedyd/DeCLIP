 python ../validate.py --arch=CLIP:ViT-L/14 --ckpt=/root/autodl-tmp/code/DeCLIP/checkpoint/20241216_211725/model_epoch_best.pth \
                    --result_folder=/root/autodl-tmp/code/DeCLIP/results --gpu_ids 0 \
                    --mask_plus_label \
                    --batch_size 16 
