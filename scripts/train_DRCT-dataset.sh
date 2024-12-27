DATASET=DRCT-2M

Prob_aug=0.5
Prob_cutmix=0.5
P_cutmixup_real_fake=0.2
P_cutmixup_real_rec=0.6
P_cutmixup_real_real=0.2

echo "P_cutmixup_real_fake: $P_cutmixup_real_fake"
echo "P_cutmixup_real_rec: $P_cutmixup_real_rec"
echo "P_cutmixup_real_real: $P_cutmixup_real_real"

SAVE_PATH=/root/autodl-tmp/code/DeCLIP/checkpoint/V2/

EXP_NAME=$(date +"%Y%m%d")

python ../train_DRCT-dataset.py --name $EXP_NAME --train_dataset $DATASET --feature_layer layer20 --fix_backbone \
                --root_path /root/autodl-tmp/AIGC_data/MSCOCO/train2017 \
                --fake_root_path /root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-inpainting/train2017,/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-v1-4/train2017 \
                --fake_indexes 2 \
                --input_size 224 \
                --is_crop \
                --mask_plus_label \
                --checkpoints_dir $SAVE_PATH \
<<<<<<< HEAD
                --gpu_ids 0,1,2 \
                --batch_size 32 --lr 0.0005 --lovasz_weight 0.1 --data_aug drct \
=======
                --gpu_ids 0,1 \
                --batch_size 16 \
                --lr 0.001 \
                --optim adam \
                --lovasz_weight 0.25 \
                --data_aug drct \
>>>>>>> 237d3af5 (修改为DRCT数据集)
                --prob_aug 0.5 \
                --prob_cutmix 0.5 \
                --prob_cutmixup_real_fake ${P_cutmixup_real_fake} \
                --prob_cutmixup_real_rec  ${P_cutmixup_real_rec} \
                --prob_cutmixup_real_real ${P_cutmixup_real_real} \
<<<<<<< HEAD
=======
                --visualize_masks \
>>>>>>> 237d3af5 (修改为DRCT数据集)
| tee ../checkpoint/log_DRCT-dataset.txt