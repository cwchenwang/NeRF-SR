dataset="fern"
W=504
H=378
batch_size=1
downscale=2
python test_refine.py --name llff-refine-${dataset}-${H}x${W}-ni-dp-ds${downscale} \
    --dataset_mode llff_refine --dataset_root /mnt/nas/raid/10017/nerf_data/nerf_llff_data/${dataset} \
    --checkpoints_dir ./checkpoints/nerf-sr-refine/ --summary_dir ./logs/nerf-sr-refine --results_dir ./results/nerf-sr-refine \
    --img_wh $W $H --batch_size $batch_size \
    --model refine --test_split test_train \
    --test_split test --refine_network maxpoolingmodel --load_epoch 3 \
    --syn_dataroot ./checkpoints/nerf-sr/llff-${dataset}-${H}x${W}-ni64-dp-ds${downscale}/30_test_vis 
