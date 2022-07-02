dataset="fern"
W=252
H=189
accelerator="dp"
extras=$1
N_importance=64
batch_size=2048
python train.py --name llff-$dataset-${H}x${W}-ni${N_importance}-${accelerator} --accelerator $accelerator $extras \
    --dataset_mode llff --dataset_root /mnt/nas/raid/10102/nerf_data/nerf_llff_data/${dataset} \
    --checkpoints_dir ./checkpoints/vanilla-nerf --summary_dir ./logs/vanilla-nerf \
    --img_wh $W $H --batch_size $batch_size \
    --n_epochs 30 --n_epochs_decay 10 \
    --print_freq 100 --vis_freq 1000 --val_freq 1000 --vis_epoch_freq 30 --val_epoch_freq 30 --save_epoch_freq 10 \
    --model nerf --N_coarse 64 --N_importance $N_importance \
    --lr_policy exp --sigma_activation relu --lr 5e-4 --lr_final 5e-6
    