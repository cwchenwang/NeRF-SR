dataset="lego"
W=200
H=200
downscale=2
N_importance=64
batch_size=2048
python test.py --name blender-down${downscale}-$dataset-${H}x${W}-ni${N_importance}-dp-ds${downscale} \
    --dataset_mode blender_downX --dataset_root /mnt/nas/raid/10017/nerf_data/nerf_synthetic/${dataset} --test_split test \
    --checkpoints_dir ./checkpoints/nerf-sr --summary_dir ./logs/nerf-sr --results_dir ./results/nerf-sr \
    --img_wh $W $H --batch_size $batch_size \
    --model nerf_downX --N_coarse 64 --N_importance $N_importance --load_epoch 20
