import os
import re
import glob
from numpy.lib.shape_base import expand_dims
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
from options import Configurable


class BaseModel(ABC, Configurable):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.isTrain, self.isTest, self.isInfer = opt.isTrain, opt.isTest, opt.isInfer
        self.device = opt.device  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir

        # losses
        self.train_loss_names = []
        self.val_loss_names = []
        self.test_loss_names = []

        # visualizations
        self.train_visual_names = []
        self.val_visual_names = []
        self.test_visual_names = []
        self.infer_visual_names = []

        # models
        self.model_names = []

        self.optimizers = []

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        
        current_epoch = 0

        if self.isTrain and opt.init_weights:
            init_weights_name, init_weights_epoch = opt.init_weights.split(':')
            self.load_networks(init_weights_name, init_weights_epoch, opt.init_weights_keys)
        
        if not self.isTrain or opt.continue_train:
            if opt.load_epoch == 'latest':
                current_epoch = max([int(os.path.basename(x).split('_')[0]) for x in glob.glob(os.path.join(self.save_dir, '*.pth')) if 'latest' not in x])
            else:
                current_epoch = int(opt.load_epoch)
            self.load_networks(opt.name, opt.load_epoch)

        if self.isTrain and opt.fix_layers:
            for name in self.model_names:
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.distributed.DistributedDataParallel):
                    net = net.module
                for param_name, params in net.named_parameters():
                    if re.match(opt.fix_layers, param_name):
                        params.requires_grad = False

        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt, last_epoch=current_epoch - 1) for optimizer in self.optimizers]

        if opt.is_master:
            self.print_networks(opt.verbose)
        return current_epoch

    def train(self):
        """Make models train mode during training time"""
        self.optimization = True
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def eval(self):
        """Make models eval mode during test time"""
        self.optimization = False
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    @abstractmethod
    def validate_iter(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        pass

    @abstractmethod
    def inference(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        pass

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
    
    def get_learning_rate(self):
        lr = self.optimizers[0].param_groups[0]['lr']
        return lr

    def get_current_visuals(self, mode):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in getattr(self, f"{mode}_visual_names"):
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self, mode):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in getattr(self, f"{mode}_loss_names"):
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.distributed.DistributedDataParallel):
                    net = net.module
                torch.save(net.cpu().state_dict(), save_path)
                net.to(self.device)

    def load_networks(self, exp_name, epoch, keys=None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.opt.checkpoints_dir, exp_name, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel) or isinstance(net, torch.nn.parallel.distributed.DistributedDataParallel):
                    net = net.module
                print('loading the model from %s' % load_path, 'with keys', keys)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                if keys is None:
                    state_dict = torch.load(load_path, map_location=self.device)
                    net.load_state_dict(state_dict)
                else:
                    state_dict = {k: v for k, v in torch.load(load_path, map_location=self.device).items() if re.match(keys, k)}
                    net.load_state_dict(state_dict, strict=False)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
