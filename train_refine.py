"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
from options.base_options import add_dist_options
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils.visualizer import save_visuals, create_writer
from utils.distributed import setup_env, cleanup_env
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse


def main(rank):
    opt = TrainOptions().parse(rank)   # get training options
    setup_env(opt)
    # dataset = create_dataset(opt, mode=opt.train_split, shuffle=True)  # create a dataset given opt.dataset_mode and other options
    # dataset_size = len(dataset) if opt.keep_last else len(dataset) - len(dataset) % opt.batch_size    # get the number of images in the dataset.
    dataset_val = create_dataset(opt, mode=opt.val_epoch_split, shuffle=False)
    dataset_iterval = create_dataset(opt, mode=opt.val_split, shuffle=False)
    iter_val = iter(dataset_iterval)
    # dataset_test = create_dataset(opt, mode=opt.test_split, shuffle=False)

    dataset = create_dataset(opt, mode='train', shuffle=True)
    dataset_size = len(dataset) if opt.keep_last else len(dataset) - len(dataset) % opt.batch_size

    if opt.is_master:
        print('The number of training data = %d' % dataset_size)
        
    model = create_model(opt)      # create a model given opt.model and other options
    current_epoch = model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = current_epoch * len(dataset.dataloader)      # the total number of training iterations
    writer = create_writer(opt)
    for epoch in range(current_epoch + 1, opt.n_epochs + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        if opt.is_master:
            writer.add_scalar('lr', model.get_learning_rate(), global_step=epoch)
            print('Learning rate:', f"{model.get_learning_rate():.3e}")
        """
        # add these lines to shuffle the dataset again before each epoch
        if opt.accelerator == 'ddp':
            dataset.dataloader.sampler.set_epoch(epoch)
        """
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += 1
            epoch_iter += 1

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if opt.is_master and total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses('train')
                t_comp = time.time() - iter_start_time
                for loss_name, loss_val in losses.items():
                    writer.add_scalars(f"{loss_name}", {'train': loss_val}, global_step=total_iters)
                print(f"Epoch {epoch} - Iteration {epoch_iter}/{len(dataset.dataloader)} (comp time {t_comp:.3f}, data time {t_data:.3f})")
                print("Training losses |", ' '.join([f"{k}: {v:.3e}" for k, v in losses.items()]))

            if opt.is_master and total_iters % opt.vis_freq == 0:
                save_visuals(os.path.join(model.save_dir, f"vis"), model.get_current_visuals('train'), total_iters)
            
            if opt.is_master and total_iters % opt.val_freq == 0:
                model.eval()
                try:
                    val_data = next(iter_val)
                except StopIteration:
                    iter_val = iter(dataset_iterval)
                    val_data = next(iter_val)
                with torch.no_grad():
                    model.set_input(val_data)
                    model.validate_iter()
                model.train()
                val_losses = model.get_current_losses('val_iter')
                for loss_name, loss_val in val_losses.items():
                    writer.add_scalars(f"{loss_name}", {'val': loss_val}, global_step=total_iters)
                if total_iters % opt.vis_freq == 0:   # display images
                    save_visuals(os.path.join(model.save_dir, 'vis'), model.get_current_visuals('val_iter'), total_iters)
                print("Validation iter losses |", ' '.join([f"{k}: {v:.3e}" for k, v in val_losses.items()]))

        if opt.is_master and epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('Saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
            model.save_networks('latest')

        if opt.is_master and epoch % opt.val_epoch_freq == 0:
            model.eval()
            with torch.no_grad():
                model.validate(dataset_val)
            val_losses = model.get_current_losses('val')
            for loss_name, loss_val in val_losses.items():
                writer.add_scalars(f"{loss_name}", {'val_full': loss_val}, global_step=total_iters)
            save_visuals(os.path.join(model.save_dir, f"{epoch}_val_vis"), model.get_current_visuals('val'))
            print("Validation losses |", ' '.join([f"{k}: {v:.3e}" for k, v in val_losses.items()]))
    
        if opt.is_master:
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))

        model.update_learning_rate()    # update learning rates in the beginning of every epoch.


        # may cause program to stuck
        if opt.accelerator == 'ddp':
            dist.barrier()

    cleanup_env(opt)


def run_dp():
    main(None)

def run_ddp():
    n_gpus = torch.cuda.device_count()
    mp.spawn(main, nprocs=n_gpus, join=True)


if __name__ == '__main__':
    parser = add_dist_options(argparse.ArgumentParser())
    opt, _ = parser.parse_known_args()
    if opt.accelerator == 'dp':
        run_dp()
    elif opt.accelerator == 'ddp':
        run_ddp()
