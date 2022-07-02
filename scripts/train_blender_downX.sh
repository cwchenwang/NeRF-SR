dataset="lego"
W=200
H=200
accelerator="dp"
downscale=2
N_importance=64
batch_size=512
python train_blender.py --name blender-down${downscale}-$dataset-${H}x${W}-ni${N_importance}-${accelerator}-ds${downscale} --accelerator $accelerator \
    --dataset_mode blender_downX --dataset_root /mnt/nas/raid/10017/nerf_data/nerf_synthetic/${dataset} --val_epoch_split test \
    --checkpoints_dir ./checkpoints/nerf-sr --summary_dir ./logs/nerf-sr \
    --img_wh $W $H --batch_size $batch_size \
    --n_epochs 20 --n_epochs_decay 10 \
    --print_freq 100 --vis_freq 1000 --val_freq 1000 --val_epoch_freq 20 --vis_epoch_freq 20 --save_epoch_freq 5 \
    --model nerf_downX --N_coarse 64 --N_importance $N_importance \
    --lr_policy exp --sigma_activation relu --lr 5e-4 --lr_final 5e-6 \
    --downscale ${downscale}