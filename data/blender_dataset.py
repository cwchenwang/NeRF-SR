import math
import torch
import numpy as np
from data.base_dataset import BaseDataset
import json
import os
from PIL import Image
from torchvision import transforms as T
from models.utils import *


class BlenderDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.set_defaults(white_bkgd=True, noise_std=0.)
        return parser    
        
    def __init__(self, opt, mode):
        self.opt = opt
        self.mode = mode
        assert self.mode in ['train', 'train_crop', 'val', 'test']

        self.root_dir = opt.dataset_root
        self.split = mode
        self.img_wh = opt.img_wh
        assert self.img_wh[0] == self.img_wh[1], 'image width must equal image height!'

        self.define_transforms()

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        self.split_path = self.split
        if self.split == 'train_crop':
            self.split_path = 'train'
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split_path}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                                                                     # when W=800

        self.focal *= self.img_wh[0]/800 # modify focal length to match size self.img_wh

        # bounds, common for all scenes
        self.near = 2.0
        self.far = 6.0
        self.bounds = np.array([self.near, self.far])
        
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(h, w, self.focal, self.opt.use_pixel_centers) # (h, w, 3)
            
        if self.split == 'train' or self.split == 'train_crop': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (4, h, w)
                if img.shape[0] == 4:
                    img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                    img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                else:
                    img = img.view(3, -1).permute(1, 0)
               
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)

                if self.split == 'train_crop':
                    dH = int(self.img_wh[1]//2 * self.opt.precrop_frac)
                    dW = int(self.img_wh[0]//2 * self.opt.precrop_frac)
                    def center_crop(data):
                        data = data.view(self.img_wh[1], self.img_wh[0], -1)
                        data = data[self.img_wh[1]//2 - dH: self.img_wh[1]//2 + dH, self.img_wh[0]//2 - dW: self.img_wh[0]//2 + dW, :]
                        data = data.reshape(2*dH * 2*dW, -1)
                        return data
                    img = center_crop(img)
                    rays_o = center_crop(rays_o)
                    rays_d = center_crop(rays_d)
                
                self.all_rgbs += [img]
                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1]),
                                             rays_d],
                                             1)] # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w, 3)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train' or self.split == 'train_crop':
            return len(self.all_rays)
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.meta['frames'])

    def __getitem__(self, idx):
        if self.split == 'train' or self.split == 'train_crop': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (4, H, W)
            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d, 
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1]),
                              rays_d],
                              1) # (H*W, 8)

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w,
                      'valid_mask': valid_mask}

        return sample