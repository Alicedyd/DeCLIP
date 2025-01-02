 CHECK_POINT_PATH=/root/autodl-tmp/code/DeCLIP/checkpoint/V2/20241230-unet/model_best_epoch_4_acc_93.36658494660016.pth
 RESULT_PATH=V2/20241230-unet
 
 python ../validate.py --arch=CLIP:ViT-L/14 --ckpt=$CHECK_POINT_PATH \
                    --result_folder=/root/autodl-tmp/code/DeCLIP/results/$RESULT_PATH --gpu_ids 0 \
                    --mask_plus_label \
                    --batch_size 16  --visualize_masks --unet\