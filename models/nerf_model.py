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
from models import find_network_using_name
from .base_model import BaseModel
from .networks import init_net
from .embedding import BaseEmbedding
from utils.utils import chunk_batch, find_class_using_name
from utils.visualizer import Visualizee, depth2im
from tqdm import tqdm
import itertools
from options import get_option_setter, str2bool
from .rendering import VolumetricRenderer
from .utils import *
from .criterions import *


class NeRFModel(BaseModel):
    """
    TODO:
    - SSIM calculation
    """
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--mlp_network', type=str, default='vanilla_mlp')
        parser.add_argument('--embedding', type=str, default='positional_encoding')
        parser.add_argument('--N_coarse', type=int, default=64)
        parser.add_argument('--N_importance', type=int, default=64)
        parser.add_argument('--lindisp', action='store_true')
        parser.add_argument('--noise_std', type=float, default=0., help="std dev of noise added to regularize sigma output. (used in the llff dataset only)")
        parser.add_argument('--white_bkgd', action='store_true', help="using white color as default background. (used in the blender dataset only)")
        parser.add_argument('--randomized', type=str2bool, default=True)

        parser.add_argument('--dim_rgb', type=int, default=3)
        parser.add_argument('--dim_pos', type=int, default=3)
        parser.add_argument('--dim_dir', type=int, default=3)
        parser.add_argument('--deg_pos', type=int, default=10)
        parser.add_argument('--deg_dir', type=int, default=4)

        parser.add_argument('--lambda_coarse_mse', type=float, default=1.)
        parser.add_argument('--lambda_fine_mse', type=float, default=1.)
        parser.add_argument('--lambda_coarse_depth_lap', type=float, default=0.)
        parser.add_argument('--lambda_fine_depth_lap', type=float, default=0.)
        parser.add_argument('--lambda_coarse_vgg', type=float, default=0.)
        parser.add_argument('--lambda_fine_vgg', type=float, default=0.)

        parser.add_argument('--color_activation', type=str, default='sigmoid', choices=['none', 'sigmoid'])
        parser.add_argument('--sigma_activation', type=str, default='relu', choices=['relu', 'softplus'])

        parser.add_argument('--bilateral_gamma', type=float, default=0.1)
        parser.add_argument('--with_ref', action='store_true')
        parser.add_argument('--no_ref_loss', action='store_true', help='whether to use reference image loss')
        parser.add_argument('--downscale', type=int, default=2)

        opt, _ = parser.parse_known_args()
        embedding_option_setter = get_option_setter(find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding))
        parser = embedding_option_setter(parser)

        opt, _ = parser.parse_known_args()
        for key, network_name in opt.__dict__.items():
            if key.endswith('_network'):
                network_option_setter = get_option_setter(find_class_using_name('models.networks', network_name, type=nn.Module))
                parser = network_option_setter(parser)

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
        self.train_loss_names = ['coarse_mse', 'fine_mse', 'coarse_psnr', 'fine_psnr', 'tot']
        self.val_iter_loss_names = ['coarse_mse', 'fine_mse', 'coarse_psnr', 'fine_psnr', 'tot']
        self.val_loss_names = ['coarse_psnr', 'fine_psnr']
        self.test_loss_names = []

        if opt.lambda_coarse_depth_lap > 0 and opt.patch_size > 2:
            self.train_loss_names += ['coarse_depth_lap']
        if opt.lambda_fine_depth_lap > 0 and opt.patch_size > 2:
            self.train_loss_names += ['fine_depth_lap']
        
        if opt.lambda_coarse_vgg > 0 and opt.patch_size >= 32:
            self.train_loss_names += ['coarse_vgg']
        if opt.lambda_fine_vgg > 0 and opt.patch_size >= 32:
            self.train_loss_names += ['fine_vgg']
        
        if self.opt.with_ref and not self.opt.no_ref_loss:
            self.train_loss_names.extend(['ref_coarse_mse', 'ref_fine_mse'])
            self.val_iter_loss_names.extend(['ref_coarse_mse', 'ref_fine_mse'])

        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.train_visual_names = []
        self.val_iter_visual_names = ['coarse_pred', 'fine_pred']
        self.val_visual_names = ['coarse_pred', 'fine_pred']
        self.test_visual_names = ['coarse_pred', 'fine_pred', 'coarse_pred_gif', 'fine_pred_gif']
        self.infer_visual_names = ['coarse_pred', 'fine_pred']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['Coarse', 'Fine']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.learnable_codes = {}
        self.fixed_codes = {}
        self.netCoarse = init_net(find_network_using_name(opt.mlp_network)(opt), opt)
        self.netFine = init_net(find_network_using_name(opt.mlp_network)(opt), opt)
        self.embeddings = {
            'pos': find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_pos, opt.deg_pos, opt),
            'dir': find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_dir, opt.deg_dir, opt)
        }
        self.models = {
            'coarse': self.netCoarse,
            'fine': self.netFine
        }
        self.losses = {
            'mse': ColorMSELoss(opt),
            'psnr': PSNR(opt),
            'lap': BilateralLaplacianLoss(opt),
            'vgg': VGGPerceptualLoss(opt)
        }
        self.renderer = VolumetricRenderer(self.opt)
        self.randomized = opt.randomized
        if self.isTrain:  # only defined during training time
            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer = torch.optim.Adam([{
                'params': itertools.chain(self.netCoarse.parameters(), self.netFine.parameters()),
                'initial_lr': opt.lr
            }], lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]


        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # data_rays: [N, 8]
        # data_rgbs: [N, 3]
        """
        x.ndim == 3 when
        1. [train with patch_size > 1] (N x P**2 x D)
        2. [val/test] (1 x H*W x D)
        """
        pack = lambda x: x.view(-1, x.shape[-1]) if x.ndim == 3 else x
        for name, v in input.items():
            setattr(self, f"data_{name}", pack(v).to(self.device))
    
    def train(self):
        super().train()
        self.randomized = self.opt.randomized
        self.H, self.W = self.opt.patch_size, self.opt.patch_size

    def eval(self):
        super().eval()
        self.randomized = False
        self.H, self.W = self.opt.img_wh[1], self.opt.img_wh[0]

    def render_rays(self, model, xyz, dir_embedded, **kwargs):
        N_rays = xyz.shape[0]
        N_samples = xyz.shape[1]
        xyz = xyz.view(-1, self.opt.dim_pos)
        B = xyz.shape[0]
        xyz_embedded = self.embeddings['pos'](xyz)
        """
        THIS IS WRONG:
        dir_embedded = dir_embedded.repeat(N_samples, 1)
        THESE ARE RIGHT:
        dir_embedded = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples)
        dir_embedded = dir_embedded[:,None,...].expand(-1, N_samples, -1).reshape(N_rays * N_samples, -1)
        """
        dir_embedded = dir_embedded[:,None,...].expand(-1, N_samples, -1).reshape(N_rays * N_samples, -1)
        xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], dim=-1)
        out = chunk_batch(model, self.opt.point_chunk, xyzdir_embedded, **kwargs)
        out = out.view(N_rays, N_samples, -1)
        out_rgbs = out[..., :self.opt.dim_rgb]
        out_sigmas = out[..., self.opt.dim_rgb]
        return out_rgbs, out_sigmas

    def forward_rays(self, rays):
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
        self.near, self.far = near[0].cpu().numpy(), far[0].cpu().numpy()
        # TODO: dir is not normalized in LLFF light field data
        dir_embedded = self.embeddings['dir'](rays[:, 8:11]) # (N_rays, embed_dir_channels)
        z_vals, xyz_coarse = sample_along_rays(rays_o, rays_d, near, far, self.opt.N_coarse, self.randomized, self.opt.lindisp)

        coarse_rgbs, coarse_sigmas = self.render_rays(self.models['coarse'], xyz_coarse, dir_embedded, sigma_only=False)
        coarse_sigmas = add_gaussian_noise(coarse_sigmas, self.randomized, self.opt.noise_std)
        coarse_comp_rgbs, coarse_depth, coarse_opacity, coarse_weights = self.renderer(coarse_rgbs, coarse_sigmas, z_vals, self.opt.white_bkgd)

        out = {
            'coarse_comp_rgbs': coarse_comp_rgbs,
            'coarse_depth': coarse_depth,
            'coarse_opacity': coarse_opacity,
            'coarse_weights': coarse_weights
        }

        if self.opt.N_importance > 0: # sample points for fine model
            # detach so that grad doesn't propogate to weights_coarse from here
            z_vals, xyz_fine = resample_along_rays(rays_o, rays_d, z_vals, coarse_weights.detach(), self.opt.N_importance, self.randomized)
            fine_rgbs, fine_sigmas = self.render_rays(self.models['fine'], xyz_fine, dir_embedded, sigma_only=False)
            fine_sigmas = add_gaussian_noise(fine_sigmas, self.randomized, self.opt.noise_std)
            fine_comp_rgbs, fine_depth, fine_opacity, fine_weights = self.renderer(fine_rgbs, fine_sigmas, z_vals, self.opt.white_bkgd)
            out.update({
                'fine_comp_rgbs': fine_comp_rgbs,
                'fine_depth': fine_depth,
                'fine_opacity': fine_opacity,
                'fine_weights': fine_weights
            })

        return out


    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        out = chunk_batch(self.forward_rays, self.opt.ray_chunk, self.data_rays)
        for name, v in out.items():
            setattr(self, f"out_{name}", v)
        if self.opt.with_ref and hasattr(self, 'data_ref_rays'):
            out_ref = chunk_batch(self.forward_rays, self.opt.ray_chunk, self.data_ref_rays)
            for name, v in out_ref.items():
                setattr(self, f"out_ref_{name}", v) 
    
    def calculate_losses(self):
        self.loss_tot = 0

        # coarse mse
        self.loss_coarse_mse = self.losses['mse'](self.out_coarse_comp_rgbs, self.data_rgbs) * self.opt.lambda_coarse_mse
        self.loss_tot += self.loss_coarse_mse

        # fine mse
        if hasattr(self, 'out_fine_comp_rgbs'):
            self.loss_fine_mse = self.losses['mse'](self.out_fine_comp_rgbs, self.data_rgbs) * self.opt.lambda_fine_mse
        else:
            self.loss_fine_mse = 0
        self.loss_tot += self.loss_fine_mse

        if self.opt.with_ref and hasattr(self, 'data_ref_rays'):
            self.loss_ref_coarse_mse = self.losses['mse'](self.out_ref_coarse_comp_rgbs, self.data_ref_rgbs) / (self.opt.downscale**2)
            self.loss_ref_fine_mse = self.losses['mse'](self.out_ref_fine_comp_rgbs, self.data_ref_rgbs) / (self.opt.downscale**2)
            self.loss_tot += self.loss_ref_coarse_mse + self.loss_ref_fine_mse
        else:
            self.loss_ref_coarse_mse = 0
            self.loss_ref_fine_mse = 0

        # coarse depth
        if self.opt.lambda_coarse_depth_lap > 0 and self.opt.patch_size > 2:
            self.loss_coarse_depth_lap = self.losses['lap'](
                self.out_coarse_depth.view(-1, self.H, self.W),
                self.data_rgbs.view(-1, self.H, self.W, self.opt.dim_rgb)
            ) * self.opt.lambda_coarse_depth_lap
            self.loss_tot += self.loss_coarse_depth_lap

        # fine depth
        if self.opt.lambda_fine_depth_lap > 0 and self.opt.patch_size > 2:
            if hasattr(self, 'out_fine_depth'):
                self.loss_fine_depth_lap = self.losses['lap'](
                    self.out_fine_depth.view(-1, self.H, self.W),
                    self.data_rgbs.view(-1, self.H, self.W, self.opt.dim_rgb)
                ) * self.opt.lambda_fine_depth_lap
                self.loss_tot += self.loss_fine_depth_lap

        if self.opt.lambda_coarse_vgg > 0 and self.opt.patch_size >= 32:
            self.loss_coarse_vgg = self.losses['vgg'](
                self.out_coarse_comp_rgbs.reshape(-1, self.H, self.W, self.opt.dim_rgb).permute(0, 3, 1, 2).contiguous(),
                self.data_rgbs.reshape(-1, self.H, self.W, self.opt.dim_rgb).permute(0, 3, 1, 2).contiguous()
            ) * self.opt.lambda_coarse_vgg
            self.loss_tot += self.loss_coarse_vgg

        if self.opt.lambda_fine_vgg > 0 and self.opt.patch_size >= 32:
            self.loss_fine_vgg = self.losses['vgg'](
                self.out_fine_comp_rgbs.reshape(-1, self.H, self.W, self.opt.dim_rgb).permute(0, 3, 1, 2).contiguous(),
                self.data_rgbs.reshape(-1, self.H, self.W, self.opt.dim_rgb).permute(0, 3, 1, 2).contiguous()
            ) * self.opt.lambda_fine_vgg
            self.loss_tot += self.loss_coarse_vgg            

        with torch.no_grad():
            self.loss_coarse_psnr = self.losses['psnr'](self.out_coarse_comp_rgbs, self.data_rgbs)
            if hasattr(self, 'out_fine_comp_rgbs'):
                self.loss_fine_psnr = self.losses['psnr'](self.out_fine_comp_rgbs, self.data_rgbs)
            else:
                self.loss_fine_psnr = 0

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        # todo
        self.calculate_losses()
        self.loss_tot.backward()       # calculate gradients of network G w.r.t. loss_G

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        if self.opt.grad_clip_val > 0:
            if self.opt.grad_clip_type == 'norm':
                nn.utils.clip_grad_norm_(itertools.chain(self.netCoarse.parameters(), self.netFine.parameters()), self.opt.grad_clip_val)
            elif self.opt.grad_clip_type == 'value':
                nn.utils.clip_grad_value_(itertools.chain(self.netCoarse.parameters(), self.netFine.parameters()), self.opt.grad_clip_val)
        self.optimizer.step()        # update gradients for network G

    def calculate_vis(self, with_gt):
        W, H = self.opt.img_wh
        coarse_img = self.out_coarse_comp_rgbs.view(H, W, 3).cpu()
        fine_img = self.out_fine_comp_rgbs.view(H, W, 3).cpu()
        coarse_depth = depth2im(self.out_coarse_depth.view(H, W), self.near, self.far)
        fine_depth = depth2im(self.out_fine_depth.view(H, W), self.near, self.far)
        if with_gt:
            gt_img = self.data_rgbs.view(H, W, 3).cpu()
            coarse_pred = torch.cat([coarse_img, gt_img, coarse_depth], dim=1)
            fine_pred = torch.cat([fine_img, gt_img, fine_depth], dim=1)
        else:
            coarse_pred = torch.cat([coarse_img, coarse_depth], dim=1)
            fine_pred = torch.cat([fine_img, fine_depth], dim=1)
        self.coarse_pred_img, self.fine_pred_img = coarse_pred, fine_pred

    def validate_iter(self):
        self.forward()
        self.calculate_losses()
        self.calculate_vis(with_gt=True)
        self.coarse_pred = Visualizee('image', self.coarse_pred_img, timestamp=True, name='coarse', data_format='HWC', range=(0, 1), img_format='png')
        self.fine_pred = Visualizee('image', self.fine_pred_img, timestamp=True, name='fine', data_format='HWC', range=(0, 1), img_format='png')
            
    def validate(self, dataset):
        coarse_psnr, fine_psnr = 0, 0
        coarse_pred, fine_pred = [], []
        for i, data in enumerate(tqdm(dataset, desc="Validation", total=len(dataset.dataloader))):
            self.set_input(data)
            self.forward()
            self.calculate_losses()
            coarse_psnr += self.loss_coarse_psnr.item()
            fine_psnr += self.loss_fine_psnr.item()
            self.calculate_vis(with_gt=True)
            coarse_pred.append(
                Visualizee('image', self.coarse_pred_img, timestamp=False, name=f'{i}-coarse', data_format='HWC', range=(0, 1), img_format='png')
            )
            fine_pred.append(
                Visualizee('image', self.fine_pred_img, timestamp=False, name=f'{i}-fine', data_format='HWC', range=(0, 1), img_format='png')
            )
        self.loss_coarse_psnr = coarse_psnr / len(dataset)
        self.loss_fine_psnr = fine_psnr / len(dataset)
        self.coarse_pred, self.fine_pred = coarse_pred, fine_pred

    def test(self, dataset):
        coarse_pred_imgs, fine_pred_imgs = [], []
        for i, data in enumerate(tqdm(dataset, desc="Testing", total=len(dataset.dataloader))):
            self.set_input(data)
            self.forward()
            self.calculate_vis(with_gt=False)
            coarse_pred_imgs.append(self.coarse_pred_img)
            fine_pred_imgs.append(self.fine_pred_img)
        self.coarse_pred, self.fine_pred = [], []
        for i, (coarse_pred_img, fine_pred_img) in enumerate(zip(coarse_pred_imgs, fine_pred_imgs)):
            self.coarse_pred.append(
                Visualizee('image', coarse_pred_img, timestamp=False, name=f'{i}-coarse', data_format='HWC', range=(0, 1), img_format='png')
            )
            self.fine_pred.append(
                Visualizee('image', fine_pred_img, timestamp=False, name=f'{i}-fine', data_format='HWC', range=(0, 1), img_format='png')
            )
        self.coarse_pred_gif = Visualizee('gif', coarse_pred_imgs, timestamp=False, name=f'coarse', data_format='HWC', range=(0, 1))
        self.fine_pred_gif = Visualizee('gif', fine_pred_imgs, timestamp=False, name=f'fine', data_format='HWC', range=(0, 1))

    def inference(self, dataset):
        pass
