 CHECK_POINT_PATH=V2/20241220_165643
 
 python ../validate.py --arch=CLIP:ViT-L/14 --ckpt=/root/autodl-tmp/code/DeCLIP/checkpoint/$CHECK_POINT_PATH/model_epoch_best.pth \
                    --result_folder=/root/autodl-tmp/code/DeCLIP/results/$CHECK_POINT_PATH --gpu_ids 2 \
                    --mask_plus_label \
                    --batch_size 16 --visualize_masks\