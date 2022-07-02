dataset="fern"
W=252
H=189
N_importance=64
batch_size=2048
python test.py --name llff-${dataset}-${H}x${W}-ni${N_importance}-dp \
    --dataset_mode llff --dataset_root /mnt/nas/raid/10017/nerf_data/nerf_llff_data/${dataset} \
    --checkpoints_dir ./checkpoints/vanilla-nerf --summary_dir ./logs/vanilla-nerf --results_dir ./results/vanilla-nerf \
    --img_wh 504 378 --batch_size $batch_size \
    --model nerf --N_coarse 64 --N_importance $N_importance \
    --test_split test_train --load_epoch 30
