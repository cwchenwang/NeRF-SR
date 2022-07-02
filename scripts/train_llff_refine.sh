dataset="fern"
W=504
H=378
accelerator="dp"
downscale=2
batch_size=32

python train_refine.py --name llff-refine-$dataset-${H}x${W}-ni${N_importance}-${accelerator}-ds${downscale} --accelerator $accelerator \
    --dataset_mode llff_refine --dataset_root /mnt/nas/raid/10017/nerf_data/nerf_llff_data/${dataset} \
    --checkpoints_dir ./checkpoints/nerf-sr-refine --summary_dir ./logs/nerf-sr-refine \
    --img_wh $W $H --batch_size $batch_size \
    --n_epochs 3 --n_epochs_decay 0 \
    --print_freq 100 --vis_freq 1000 --val_freq 1000 --save_epoch_freq 1 --val_epoch_freq 1 \
    --model refine \
    --lr_policy exp --lr 5e-4 --lr_final 5e-6 \
    --syn_dataroot ./checkpoints/nerf-sr/llff-${dataset}-${H}x${W}-ni64-dp-ds${downscale}/30_val_vis \
    --refine_with_l1