import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.visualizer import save_visuals
import torch


if __name__ == '__main__':
    opt = TestOptions().parse(None)  # get test options
    # hard-code some parameters for test
    dataset = create_dataset(opt, mode=opt.test_split, shuffle=False)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    current_epoch = model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    out_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.test_split, current_epoch))  # define the website directory
    print('creating out directory', out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    model.eval()
    with torch.no_grad():
        model.test(dataset)
    save_visuals(out_dir, model.get_current_visuals('test'))