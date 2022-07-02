"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
from .base_model import BaseModel
from .networks import init_net
from utils.visualizer import Visualizee
from tqdm import tqdm


class TemplateLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
    
    def forward(self, pred, target):
        return TF.mse_loss(pred, target)  


class TemplateNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.fc1 = nn.Linear(opt.input_dim, opt.hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(opt.hidden_dim, opt.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class TemplateModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--hidden_dim', type=int, default=128)
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.train_loss_names = ['l2', 'tot']
        self.val_loss_names = ['l2', 'tot']
        self.test_loss_names = []
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.train_visual_names = []
        self.val_visual_names = []
        self.test_visual_names = []
        self.infer_visual_names = []
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['Full']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netFull = init_net(TemplateNetwork(opt), opt)
        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            self.l2Loss = TemplateLoss(opt)
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer = torch.optim.Adam([{
                'params': self.netFull.parameters(),
                'initial_lr': opt.lr
            }], lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        for name, v in input.items():
            setattr(self, f"data_{name}", v.to(self.device))


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.pred = self.netFull(self.data_x)
    

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        # todo
        self.loss_l2 = self.l2Loss(self.pred, self.data_y)
        self.loss_tot = self.loss_l2
        self.loss_tot.backward()       # calculate gradients of network G w.r.t. loss_G

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G

    def validate_iter(self):
        with torch.no_grad():
            self.forward()
            self.loss_l2 = self.l2Loss(self.pred, self.data_y)
            self.loss_tot = self.loss_l2
    
    def validate(self, dataset):
        loss_l2 = 0
        count = 0
        for i, data in enumerate(tqdm(dataset, desc="Validation", total=len(dataset.dataloader))):
            self.set_input(data)
            bs = self.data_x.shape[0]
            count += bs
            with torch.no_grad():
                self.forward()
                loss_l2 += self.l2Loss(self.pred, self.data_y)
        self.loss_l2 = loss_l2 / count
        self.loss_tot = self.loss_l2

    def test(self, dataset):
        for i, data in enumerate(tqdm(dataset, desc="Testing", total=len(dataset.dataloader))):
            self.set_input(data)
            bs = self.data_x.shape[0]
            with torch.no_grad():
                self.forward()

    def inference(self, dataset):
        pass
