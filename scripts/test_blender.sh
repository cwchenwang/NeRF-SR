dataset="lego"
W=200
H=200
extras=$1
N_importance=64
batch_size=2048
python test.py --name blender-$dataset-${H}x${W}-ni${N_importance}-dp $extras \
    --dataset_mode blender --dataset_root /mnt/nas/raid/10017/nerf_data/nerf_synthetic/${dataset} --test_split test \
    --checkpoints_dir ./checkpoints/vanilla-nerf --summary_dir ./logs/vanilla-nerf --results_dir ./results/vanilla-nerf \
    --img_wh 400 400 --batch_size $batch_size \
    --model nerf --N_coarse 64 --N_importance $N_importance --load_epoch 20
