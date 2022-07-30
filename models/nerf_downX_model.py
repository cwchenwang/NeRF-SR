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
from torch.functional import einsum
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
from .nerf_model import NeRFModel, ColorMSELoss, PSNR
from .criterions import *
import einops

class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

class NeRFDownXModel(NeRFModel):
    """
    TODO:
    - SSIM calculation
    """
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--use_var_loss', action='store_true')
        parser.add_argument('--lambda_coarse_var', type=float, default=.01)
        parser.add_argument('--lambda_fine_var', type=float, default=.01)
        parser.add_argument('--use_depth_var_loss', action='store_true')
        parser.add_argument('--lambda_coarse_depth_var', type=float, default=.01)
        parser.add_argument('--lambda_fine_depth_var', type=float, default=.01)
        parser.add_argument('--ds_method', type=str, default='lanc', choices=['avg', 'lanc'])
        parser.add_argument('--with_sr', action='store_true')
        parser.add_argument('--with_netD', action='store_true')
        parser.add_argument('--dis_network', type=str, default='nlayerdiscriminator')
        parser.add_argument('--patch_len', type=int, default=32)
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        parser.add_argument('--lambda_gan', type=float, default=0.005)
        parser.add_argument('--use_grad', action='store_true')
        parser.add_argument('--gan_lr', type=float, default=5e-4)
        parser.add_argument('--gan_mode', type=str, default='vanilla', choices=['vanilla', 'lsgan', 'wgangp'])
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--gan_iter', type=int, default=10)
        parser.add_argument('--gamma_correct', action='store_true')
        parser.add_argument('--reg_patch', action='store_true', help='render patches for regularization')
        parser.add_argument('--reg_patch_len', type=int, default=1, help='patch size for regularization')
        parser.add_argument('--reg_patch_freq', type=int, default=5, help='iteration freq for regulariza patches')
        parser.add_argument('--reg_lambda_tv', type=float, default=1.0)
        parser = NeRFModel.modify_commandline_options(parser)
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
        self.train_loss_names = ['coarse_mse', 'fine_mse', 'coarse_psnr', 'fine_psnr', 
                                    'coarse_psnr_ori', 'fine_psnr_ori', 'tot']
        self.val_iter_loss_names = ['coarse_mse', 'fine_mse', 'coarse_psnr', 'fine_psnr', 
                                    'coarse_psnr_ori', 'fine_psnr_ori', 'tot']
        if self.opt.with_ref and not self.opt.no_ref_loss:
            self.train_loss_names.extend(['ref_coarse_mse', 'ref_fine_mse'])
            self.val_iter_loss_names.extend(['ref_coarse_mse', 'ref_fine_mse'])
        if self.opt.use_var_loss:
            self.train_loss_names.extend(['out_coarse_var', 'out_fine_var'])
            self.val_iter_loss_names.extend(['out_coarse_var', 'out_fine_var'])
        if self.opt.use_depth_var_loss:
            self.train_loss_names.extend(['coarse_depth_var', 'fine_depth_var'])
            self.val_iter_loss_names.extend(['coarse_depth_var', 'fine_depth_var'])
        if self.opt.sisr_path != None:
            self.train_loss_names.extend(['coarse_mse_sr', 'fine_mse_sr'])
        if self.opt.reg_patch:
            self.train_loss_names.extend(['coarse_patch', 'fine_patch'])
        
        self.val_loss_names = ['coarse_psnr', 'fine_psnr', 'coarse_psnr_ori', 'fine_psnr_ori']
        self.test_loss_names = []
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.train_visual_names = []
        self.val_iter_visual_names = ['coarse_pred', 'fine_pred', 'coarse_pred_ori', 'fine_pred_ori']
        self.val_visual_names = ['coarse_pred', 'fine_pred', 'coarse_pred_ori', 'fine_pred_ori',
                                'coarse_depth_mats_ori', 'fine_depth_mats_ori', 'coarse_depth_mats', 'fine_depth_mats']
        self.test_visual_names = ['coarse_pred', 'fine_pred', 'coarse_pred_ori', 'fine_pred_ori', 
                                    'coarse_pred_gif', 'fine_pred_gif', 'coarse_pred_ori_gif', 'fine_pred_ori_gif',
                                'coarse_depth_mats_ori', 'fine_depth_mats_ori', 'coarse_depth_mats', 'fine_depth_mats']
        self.infer_visual_names = ['coarse_pred', 'fine_pred', 'coarse_pred_ori, fine_pred_ori']
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
            'tv': TVLoss(opt)
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
            
            if self.opt.with_netD:
                self.model_names.append('D')
                self.train_loss_names.extend(['G_GAN', 'G_L1', 'D_real', 'D_fake'])
                self.val_iter_loss_names.extend(['G_GAN', 'G_L1', 'D_real', 'D_fake'])
                self.val_iter_visual_names.extend(['fakeB', 'realB'])

                # placeholder
                self.fakeB = Visualizee('image', torch.rand((3,3,3)), timestamp=True, name='fakeB', data_format='HWC', range=(0, 1), img_format='png')
                # self.realA = Visualizee('image', torch.rand((3,3,3)), timestamp=True, name='realA', data_format='HWC', range=(0, 1), img_format='png')
                self.realB = Visualizee('image', torch.rand((3,3,3)), timestamp=True, name='realB', data_format='HWC', range=(0, 1), img_format='png')
                if self.opt.use_grad:
                    self.val_iter_visual_names.extend(['fakeB_grad', 'realB_grad'])
                    self.fakeB_grad = Visualizee('image', torch.rand((3,3,3)), timestamp=True, name='fakeB-grad', data_format='HWC', range=(0, 1), img_format='png')
                    self.realB_grad = Visualizee('image', torch.rand((3,3,3)), timestamp=True, name='realB-grad', data_format='HWC', range=(0, 1), img_format='png')

                self.losses['gan'] = GANLoss(self.opt.gan_mode).to(self.device)
                self.losses['l1'] = torch.nn.L1Loss()
                self.loss_D_fake, self.loss_D_real = 0., 0.
                self.loss_G_GAN, self.loss_G_L1 = 0., 0.
                # self.criterionGAN = GANLoss('lsgan')
                # self.criterionL1 = torch.nn.L1Loss()
                self.netD = init_net(find_network_using_name('NLayerDiscriminator')(opt), opt)
                self.optimizer_D = torch.optim.Adam([{'params': self.netD.parameters(), 'initial_lr': opt.gan_lr}], lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)


        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # data_rays: [N, 8]
        # data_rgbs: [N, 3]
        pack = lambda x: x.squeeze() if x.shape[0] == 1 else x # N = 1 when val/test/infer
        for name, v in input.items():
            v = pack(v)
            #v = v.view(v.shape[0]*v.shape[1], -1) if len(v.shape) == 3 else v
            v = v.view(-1, v.shape[-1]) if (v.ndim==3 or v.ndim==4) else v
            setattr(self, f"data_{name}", v.to(self.device))
        
    def train(self):
        super().train()
        self.randomized = self.opt.randomized
        self.H, self.W = self.opt.reg_patch_len, self.opt.reg_patch_len

    def eval(self):
        super().eval()
        self.randomized = False
        self.H, self.W = self.opt.img_wh[1]//self.opt.downscale, self.opt.img_wh[0]//self.opt.downscale

    def render_rays(self, model, xyz, dir_embedded, **kwargs):
        N_rays = xyz.shape[0]
        N_samples = xyz.shape[1]
        xyz = xyz.view(-1, self.opt.dim_pos)
        B = xyz.shape[0]
        xyz_embedded = self.embeddings['pos'](xyz)
        dir_embedded = dir_embedded.repeat_interleave(N_samples, dim=0)
        xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], dim=-1)
        out = chunk_batch(model, self.opt.point_chunk, xyzdir_embedded, **kwargs)
        out = out.view(N_rays, N_samples, -1)
        out_rgbs = out[..., :self.opt.dim_rgb]
        if self.opt.gamma_correct:
            out_rgbs_temp = torch.pow(out_rgbs, 1/2.2)
        if torch.isnan(out_rgbs).any():
            import pdb; pdb.set_trace();   
        if self.opt.gamma_correct:
            out_rgbs = out_rgbs_temp
        out_sigmas = out[..., self.opt.dim_rgb]
        return out_rgbs, out_sigmas

    def forward_rays(self, rays):
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)
        self.near, self.far = near[0].cpu().numpy(), far[0].cpu().numpy()
        # TODO: dir is not normalized in LLFF light field data
        dir_embedded = self.embeddings['dir'](rays_d) # (N_rays, embed_dir_channels)
        z_vals, xyz_coarse = sample_along_rays(rays_o, rays_d, near, far, self.opt.N_coarse, self.randomized, self.opt.lindisp)

        coarse_rgbs, coarse_sigmas = self.render_rays(self.models['coarse'], xyz_coarse, dir_embedded, sigma_only=False)
        coarse_sigmas = add_gaussian_noise(coarse_sigmas, self.randomized, self.opt.noise_std)
        coarse_comp_rgbs, corase_depth, coarse_opacity, coarse_weights = self.renderer(coarse_rgbs, coarse_sigmas, z_vals, self.opt.white_bkgd)

        out = {
            'coarse_comp_rgbs': coarse_comp_rgbs,
            'coarse_depth': corase_depth,
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
    
    def comp_low_res_output(self):
        self.out_coarse_comp_rgbs_ori = self.out_coarse_comp_rgbs.clone() 
        if not hasattr(self, 'data_rgbs'): # TODO: Hard coded
            import warnings
            warnings.warn('Hard coded')
            self.data_rgbs = torch.zeros((47628, 1))
        if self.opt.use_var_loss:
            self.loss_out_coarse_var = torch.sum(torch.var(torch.reshape(self.out_coarse_comp_rgbs, 
                                                    (self.data_rgbs.shape[0], self.opt.downscale*self.opt.downscale, -1)), dim=1))
            self.loss_out_fine_var = torch.sum(torch.var(torch.reshape(self.out_fine_comp_rgbs, 
                                                    (self.data_rgbs.shape[0], self.opt.downscale*self.opt.downscale, -1)), dim=1))
        self.out_coarse_comp_rgbs = torch.mean(torch.reshape(self.out_coarse_comp_rgbs, 
                                                (self.data_rgbs.shape[0], self.opt.downscale*self.opt.downscale, -1)), dim=1)
        self.out_coarse_depth_ori = self.out_coarse_depth.clone()
        self.out_coarse_depth = torch.mean(torch.reshape(self.out_coarse_depth, 
                                                (self.data_rgbs.shape[0], self.opt.downscale*self.opt.downscale, -1)), dim=1)
        if hasattr(self, 'out_fine_comp_rgbs'):
            self.out_fine_comp_rgbs_ori = self.out_fine_comp_rgbs.clone()
            self.out_fine_comp_rgbs = torch.mean(torch.reshape(self.out_fine_comp_rgbs, 
                                                (self.data_rgbs.shape[0], self.opt.downscale*self.opt.downscale, -1)), dim=1)
            self.out_fine_depth_ori = self.out_fine_depth.clone()
            self.out_fine_depth = torch.mean(torch.reshape(self.out_fine_depth, 
                                                (self.data_rgbs.shape[0], self.opt.downscale*self.opt.downscale, -1)), dim=1)
        if self.opt.use_depth_var_loss:
            self.loss_coarse_depth_var = torch.sum(torch.var(torch.reshape(self.out_coarse_depth_ori, 
                                                    (self.data_rgbs.shape[0], self.opt.downscale*self.opt.downscale, -1)) / self.far, dim=1))
            self.loss_fine_depth_var = torch.sum(torch.var(torch.reshape(self.out_fine_depth_ori, 
                                                    (self.data_rgbs.shape[0], self.opt.downscale*self.opt.downscale, -1)) / self.far, dim=1))

    def calculate_losses(self):
        self.comp_low_res_output()
        self.loss_coarse_mse = self.losses['mse'](self.out_coarse_comp_rgbs, self.data_rgbs) * self.opt.lambda_coarse_mse
        if hasattr(self, 'out_fine_comp_rgbs'):
            self.loss_fine_mse = self.losses['mse'](self.out_fine_comp_rgbs, self.data_rgbs) * self.opt.lambda_fine_mse
        else:
            self.loss_fine_mse = 0
        self.loss_tot = self.loss_coarse_mse + self.loss_fine_mse

        if self.opt.sisr_path != None:
            self.loss_coarse_mse_sr = self.losses['mse'](self.out_coarse_comp_rgbs_ori, self.data_rgbs_sr)
            self.loss_fine_mse_sr = self.losses['mse'](self.out_fine_comp_rgbs_ori, self.data_rgbs_sr)
            self.loss_tot += (self.loss_coarse_mse_sr + self.loss_fine_mse_sr)

        if self.opt.with_ref and hasattr(self, 'data_ref_rays'):
            self.loss_ref_coarse_mse = self.losses['mse'](self.out_ref_coarse_comp_rgbs, self.data_ref_rgbs) / (self.opt.downscale**2)
            self.loss_ref_fine_mse = self.losses['mse'](self.out_ref_fine_comp_rgbs, self.data_ref_rgbs) / (self.opt.downscale**2)
            self.loss_tot += self.loss_ref_coarse_mse + self.loss_ref_fine_mse

        if self.opt.use_var_loss:
            self.loss_tot += self.opt.lambda_coarse_var * self.loss_out_coarse_var + self.opt.lambda_fine_var * self.loss_out_fine_var
        if self.opt.use_depth_var_loss:
            self.loss_tot += self.opt.lambda_coarse_depth_var * self.loss_coarse_depth_var + \
                             self.opt.lambda_fine_depth_var * self.loss_fine_depth_var
        with torch.no_grad():
            self.loss_coarse_psnr = self.losses['psnr'](self.out_coarse_comp_rgbs, self.data_rgbs)
            if hasattr(self, 'out_fine_comp_rgbs'):
                self.loss_fine_psnr = self.losses['psnr'](self.out_fine_comp_rgbs, self.data_rgbs)
            else:
                self.loss_fine_psnr = 0
            
            if hasattr(self, 'data_rgbs_ori'):
                self.loss_coarse_psnr_ori = self.losses['psnr'](self.out_coarse_comp_rgbs_ori, self.data_rgbs_ori)
                self.loss_fine_psnr_ori = self.losses['psnr'](self.out_fine_comp_rgbs_ori, self.data_rgbs_ori)

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

    def unflatten_reshape(self, input):
        W, H = self.opt.img_wh
        W1, H1 = W // self.opt.downscale, H // self.opt.downscale
        input = input.reshape(H1*W1, self.opt.downscale*self.opt.downscale, -1)
        input = einops.rearrange(input, '(h1 w1) (s1 s2) c -> (h1 s1) (w1 s2) c',
                                  h1=H1, s1=self.opt.downscale)
        return input

    def calculate_vis(self, with_gt):
        W, H = self.opt.img_wh
        coarse_img_ori = self.unflatten_reshape(self.out_coarse_comp_rgbs_ori).cpu()
        fine_img_ori = self.unflatten_reshape(self.out_fine_comp_rgbs_ori).cpu()
        coarse_depth_ori = depth2im(self.unflatten_reshape(self.out_coarse_depth_ori), self.near, self.far)
        fine_depth_ori = depth2im(self.unflatten_reshape(self.out_fine_depth_ori), self.near, self.far)
        if with_gt:
            gt_img = self.unflatten_reshape(self.data_rgbs_ori).cpu()
            coarse_pred_ori = torch.cat([coarse_img_ori, gt_img, coarse_depth_ori], dim=1)
            fine_pred_ori = torch.cat([fine_img_ori, gt_img, fine_depth_ori], dim=1)
        else:
            coarse_pred_ori = torch.cat([coarse_img_ori, coarse_depth_ori], dim=1)
            fine_pred_ori = torch.cat([fine_img_ori, fine_depth_ori], dim=1)
        self.coarse_pred_img_ori, self.fine_pred_img_ori = coarse_pred_ori, fine_pred_ori

        W1, H1 = W // self.opt.downscale, H // self.opt.downscale
        coarse_img = self.out_coarse_comp_rgbs.view(H1, W1, 3).cpu()
        fine_img = self.out_fine_comp_rgbs.view(H1, W1, 3).cpu()
        coarse_depth = depth2im(self.out_coarse_depth.view(H1, W1), self.near, self.far)
        fine_depth = depth2im(self.out_fine_depth.view(H1, W1), self.near, self.far)
        if with_gt:
            gt_img = self.data_rgbs.view(H1, W1, 3).cpu()
            coarse_pred = torch.cat([coarse_img, gt_img, coarse_depth], dim=1)
            fine_pred = torch.cat([fine_img, gt_img, fine_depth], dim=1)
        else:
            coarse_pred = torch.cat([coarse_img, coarse_depth], dim=1)
            fine_pred = torch.cat([fine_img, fine_depth], dim=1)
        self.coarse_pred_img, self.fine_pred_img = coarse_pred, fine_pred

        self.coarse_depth_mat_ori = self.unflatten_reshape(self.out_coarse_depth_ori)
        self.fine_depth_mat_ori = self.unflatten_reshape(self.out_fine_depth_ori)
        self.coarse_depth_mat = self.out_coarse_depth.view(H1, W1)
        self.fine_depth_mat = self.out_fine_depth.view(H1, W1)

    def validate_iter(self):
        self.forward()
        self.calculate_losses()
        self.calculate_vis(with_gt=True)
        self.coarse_pred_ori = Visualizee('image', self.coarse_pred_img_ori, timestamp=True, name='coarse-ori', data_format='HWC', range=(0, 1), img_format='png')
        self.fine_pred_ori = Visualizee('image', self.fine_pred_img_ori, timestamp=True, name='fine-ori', data_format='HWC', range=(0, 1), img_format='png')
        self.coarse_pred = Visualizee('image', self.coarse_pred_img, timestamp=True, name='coarse', data_format='HWC', range=(0, 1), img_format='png')
        self.fine_pred = Visualizee('image', self.fine_pred_img, timestamp=True, name='fine', data_format='HWC', range=(0, 1), img_format='png')

        if self.opt.with_netD and hasattr(self, 'data_gan_rgbsB'):
            self.fakeB = Visualizee('image', self.out_fakeB_fine_comp_rgbs, timestamp=True, name='fakeB', data_format='HWC', range=(0, 1), img_format='png')
            # self.realA = Visualizee('image', self.data_gan_rgbs, timestamp=True, name='realA', data_format='HWC', range=(0, 1), img_format='png')
            self.realB = Visualizee('image', self.data_gan_rgbsB, timestamp=True, name='realB', data_format='HWC', range=(0, 1), img_format='png')
            if self.opt.use_grad:
                self.fakeB_grad = Visualizee('image', self.img_grad(self.out_fakeB_fine_comp_rgbs)[0], timestamp=True, name='fakeB-grad', data_format='HWC', range=(0, 1), img_format='png')
                self.realB_grad = Visualizee('image', self.img_grad(self.data_gan_rgbsB)[0], timestamp=True, name='realB-grad', data_format='HWC', range=(0, 1), img_format='png')
            
    def validate(self, dataset):
        coarse_psnr, fine_psnr = 0, 0
        coarse_psnr_ori, fine_psnr_ori = 0, 0
        coarse_pred, fine_pred = [], []
        coarse_pred_ori, fine_pred_ori = [], []
        coarse_depth_mats, fine_depth_mats = [], []
        coarse_depth_mats_ori, fine_depth_mats_ori = [], []
        for i, data in enumerate(tqdm(dataset, desc="Validation", total=len(dataset.dataloader))):
            self.set_input(data)
            self.forward()
            self.calculate_losses()
            coarse_psnr += self.loss_coarse_psnr.item()
            fine_psnr += self.loss_fine_psnr.item()
            coarse_psnr_ori += self.loss_coarse_psnr_ori.item()
            fine_psnr_ori += self.loss_fine_psnr_ori.item()
            self.calculate_vis(with_gt=True)
            coarse_pred.append(
                Visualizee('image', self.coarse_pred_img, timestamp=False, name=f'{i}-coarse', data_format='HWC', range=(0, 1), img_format='png')
            )
            fine_pred.append(
                Visualizee('image', self.fine_pred_img, timestamp=False, name=f'{i}-fine', data_format='HWC', range=(0, 1), img_format='png')
            )
            coarse_pred_ori.append(
                Visualizee('image', self.coarse_pred_img_ori, timestamp=False, name=f'{i}-coarse-ori', data_format='HWC', range=(0, 1), img_format='png')
            )
            fine_pred_ori.append(
                Visualizee('image', self.fine_pred_img_ori, timestamp=False, name=f'{i}-fine-ori', data_format='HWC', range=(0, 1), img_format='png')
            )
            coarse_depth_mats.append(
                Visualizee('matrix', self.coarse_depth_mat, timestamp=False, name=f'{i}-coarse-depth')
            )
            fine_depth_mats.append(
                Visualizee('matrix', self.fine_depth_mat, timestamp=False, name=f'{i}-fine-depth')
            )
            coarse_depth_mats_ori.append(
                Visualizee('matrix', self.coarse_depth_mat_ori, timestamp=False, name=f'{i}-coarse-depth-ori')
            )
            fine_depth_mats_ori.append(
                Visualizee('matrix', self.fine_depth_mat_ori, timestamp=False, name=f'{i}-fine-depth-ori')
            )
        self.loss_coarse_psnr = coarse_psnr / len(dataset)
        self.loss_fine_psnr = fine_psnr / len(dataset)
        self.loss_coarse_psnr_ori = coarse_psnr_ori / len(dataset)
        self.loss_fine_psnr_ori = fine_psnr_ori / len(dataset)
        self.coarse_pred, self.fine_pred = coarse_pred, fine_pred
        self.coarse_pred_ori, self.fine_pred_ori = coarse_pred_ori, fine_pred_ori
        self.coarse_depth_mats, self.fine_depth_mats = coarse_depth_mats, fine_depth_mats
        self.coarse_depth_mats_ori, self.fine_depth_mats_ori = coarse_depth_mats_ori, fine_depth_mats_ori
    
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        # fake_AB = torch.cat((self.data_gan_rgbsB, self.out_fakeB_fine_comp_rgbs), 2).permute(2, 0, 1).unsqueeze_(0)  # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_AB = self.out_fakeB_fine_comp_rgbs.permute(2, 0, 1).unsqueeze(0)
        fake_AB = self.img_grad(fake_AB) if self.opt.use_grad else fake_AB
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.losses['gan'](pred_fake, False)
        # Real
        # real_AB = torch.cat((self.data_gan_rgbsB, self.data_gan_rgbsB), 2).permute(2, 0, 1).unsqueeze_(0)
        real_AB = self.data_gan_rgbsB.permute(2, 0, 1).unsqueeze(0)
        real_AB = self.img_grad(real_AB) if self.opt.use_grad else real_AB
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.losses['gan'](pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        # fake_AB = torch.cat((self.data_gan_rgbsB, self.out_fakeB_fine_comp_rgbs), 2).permute(2, 0, 1).unsqueeze_(0)
        fake_AB = self.out_fakeB_fine_comp_rgbs.permute(2, 0, 1).unsqueeze(0)
        fake_AB = self.img_grad(fake_AB) if self.opt.use_grad else fake_AB
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.losses['gan'](pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.losses['l1'](self.out_fakeB_fine_comp_rgbs, self.data_gan_rgbsB) * self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = (self.loss_G_GAN) * self.opt.lambda_gan # omit self.loss_G_L1 since it is already done by nerf
        self.loss_G.backward()

    def img_grad(self, x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = TF.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = TF.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0
        return torch.sqrt(dx**2 + dy**2)

    def train_gan(self, input): 
        pack = lambda x: x.squeeze() if x.shape[0] == 1 else x # N = 1 when val/test/infer
        for name, v in input.items():
            v = pack(v)
            if name.startswith('gan_raysB'): # only reshape rays
                v = v.view(v.shape[0]*v.shape[1], -1) if len(v.shape) == 3 else v
            setattr(self, f"data_{name}", v.to(self.device))

        out_fakeB = chunk_batch(self.forward_rays, self.opt.ray_chunk, self.data_gan_raysB)
        for name, v in out_fakeB.items():
            setattr(self, f"out_fakeB_{name}", v)
        
        self.out_fakeB_fine_comp_rgbs = self.out_fakeB_fine_comp_rgbs.view(self.opt.patch_len, self.opt.patch_len, -1)

        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()   

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer.step()      
    
    def regularize_patch(self, input):
        pack = lambda x: x.squeeze() if x.shape[0] == 1 else x
        for name, v in input.items():
            v = pack(v)
            if name.startswith('patch_rays'):
                v = v.view(-1, v.shape[-1]) if len(v.shape) == 3 else v
            setattr(self, f"data_{name}", v.to(self.device))
        # render patches
        out_patches = chunk_batch(self.forward_rays, self.opt.ray_chunk, self.data_patch_rays)
        for name, v in out_patches.items():
            setattr(self, f"out_patch_{name}", v)
        self.out_patch_coarse_comp_rgbs = self.out_patch_coarse_comp_rgbs.view(self.opt.reg_patch_len*self.opt.downscale, self.opt.reg_patch_len*self.opt.downscale, -1)
        self.out_patch_fine_comp_rgbs = self.out_patch_fine_comp_rgbs.view(self.opt.reg_patch_len*self.opt.downscale, self.opt.reg_patch_len*self.opt.downscale, -1)

        self.optimizer.zero_grad()

        # calculate losses
        self.loss_coarse_patch = self.losses['tv'](self.out_patch_coarse_comp_rgbs)
        self.loss_fine_patch = self.losses['tv'](self.out_patch_fine_comp_rgbs)
        self.loss_patch = (self.loss_coarse_patch + self.loss_fine_patch)* self.opt.reg_lambda_tv
        self.loss_patch.backward()

        self.optimizer.step()   


    def test(self, dataset):
        coarse_pred_imgs, fine_pred_imgs = [], []
        coarse_pred_imgs_ori, fine_pred_imgs_ori = [], []
        coarse_depth_mats, fine_depth_mats = [], []
        coarse_depth_mats_ori, fine_depth_mats_ori = [], []
        for i, data in enumerate(tqdm(dataset, desc="Testing", total=len(dataset.dataloader))):
            self.set_input(data)
            self.forward()
            self.comp_low_res_output()
            self.calculate_vis(with_gt=False)
            coarse_pred_imgs.append(self.coarse_pred_img)
            fine_pred_imgs.append(self.fine_pred_img)
            coarse_pred_imgs_ori.append(self.coarse_pred_img_ori)
            fine_pred_imgs_ori.append(self.fine_pred_img_ori)
            coarse_depth_mats.append(
                Visualizee('matrix', self.coarse_depth_mat, timestamp=False, name=f'{i}-coarse-depth')
            )
            fine_depth_mats.append(
                Visualizee('matrix', self.fine_depth_mat, timestamp=False, name=f'{i}-fine-depth')
            )
            coarse_depth_mats_ori.append(
                Visualizee('matrix', self.coarse_depth_mat_ori, timestamp=False, name=f'{i}-coarse-depth-ori')
            )
            fine_depth_mats_ori.append(
                Visualizee('matrix', self.fine_depth_mat_ori, timestamp=False, name=f'{i}-fine-depth-ori')
            )
        self.coarse_pred, self.fine_pred = [], []
        self.coarse_pred_ori, self.fine_pred_ori = [], []
        for i, (coarse_pred_img, fine_pred_img) in enumerate(zip(coarse_pred_imgs, fine_pred_imgs)):
            self.coarse_pred.append(
                Visualizee('image', coarse_pred_img, timestamp=False, name=f'{i}-coarse', data_format='HWC', range=(0, 1), img_format='png')
            )
            self.fine_pred.append(
                Visualizee('image', fine_pred_img, timestamp=False, name=f'{i}-fine', data_format='HWC', range=(0, 1), img_format='png')
            )
        self.coarse_pred_gif = Visualizee('gif', coarse_pred_imgs, timestamp=False, name=f'coarse', data_format='HWC', range=(0, 1))
        self.fine_pred_gif = Visualizee('gif', fine_pred_imgs, timestamp=False, name=f'fine', data_format='HWC', range=(0, 1))

        for i, (coarse_pred_img_ori, fine_pred_img_ori) in enumerate(zip(coarse_pred_imgs_ori, fine_pred_imgs_ori)):
            self.coarse_pred.append(
                Visualizee('image', coarse_pred_img_ori, timestamp=False, name=f'{i}-coarse-ori', data_format='HWC', range=(0, 1), img_format='png')
            )
            self.fine_pred.append(
                Visualizee('image', fine_pred_img_ori, timestamp=False, name=f'{i}-fine-ori', data_format='HWC', range=(0, 1), img_format='png')
            )
        self.coarse_depth_mats, self.fine_depth_mats = coarse_depth_mats, fine_depth_mats
        self.coarse_depth_mats_ori, self.fine_depth_mats_ori = coarse_depth_mats_ori, fine_depth_mats_ori
        self.coarse_pred_ori_gif = Visualizee('gif', coarse_pred_imgs_ori, timestamp=False, name=f'coarse-ori', data_format='HWC', range=(0, 1))
        self.fine_pred_ori_gif = Visualizee('gif', fine_pred_imgs_ori, timestamp=False, name=f'fine-ori', data_format='HWC', range=(0, 1))

    def inference(self, dataset):
        pass
