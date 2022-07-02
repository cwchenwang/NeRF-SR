W=504
H=378
accelerator="dp"
downscale=2
N_importance=64
# if downscale=4, change batchsize=128
batch_size=512 
dataset='fern'

python train.py --name llff-${dataset}-${H}x${W}-ni${N_importance}-${accelerator}-ds${downscale} --accelerator $accelerator \
--dataset_mode llff_downX --dataset_root /mnt/nas/raid/10017/nerf_data/nerf_llff_data/${dataset} \
--checkpoints_dir ./checkpoints/nerf-sr --summary_dir ./logs/nerf-sr \
--img_wh $W $H --batch_size $batch_size \
--n_epochs 30 --n_epochs_decay 10 \
--print_freq 100 --vis_freq 1000 --val_freq 1000 --vis_epoch_freq 30 --val_epoch_freq 30 --save_epoch_freq 10 \
--model nerf_downX --N_coarse 64 --N_importance $N_importance \
--lr_policy exp --sigma_activation relu --lr 5e-4 --lr_final 5e-6 \
--include_var --downscale ${downscale}

    