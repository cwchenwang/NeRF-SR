import math
import torch
import numpy as np
from data.base_dataset import BaseDataset
import json
import os
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
from models.utils import *
import einops

def get_random_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length
        use_pixel_corners:
            If True, generate rays through the center of each pixel. Note: While
            this is the correct way to handle rays, it is not the way rays are
            handled in the original NeRF paper. Setting this TRUE yields ~ +1 PSNR
            compared to Vanilla NeRF.
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
        indexing='xy'
    )
    i += np.random.rand(H, W)
    j += np.random.rand(H, W)
    i, j = torch.from_numpy(i), torch.from_numpy(j)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions

'''
Not adapted yet
Downscale origin images by a factor of X
'''
class BlenderDownXDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--rand_dir', action='store_true', help='sample direction randomly')
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
        self.split_path = self.split if not (self.split == 'train_crop') else 'train'
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
        self.directions =  get_random_ray_directions(h, w, self.focal) if self.opt.rand_dir else \
            get_ray_directions(h, w, self.focal, self.opt.use_pixel_centers) # (h, w, 3)
            
        if self.split == 'train' or self.split == 'train_crop': # create buffer of all rays and rgb data
            self.image_paths = []
            self.poses = []
            self.all_rays = []
            self.all_rgbs = []
            self.all_rgbs_ori = []
            for frame in self.meta['frames']:
                pose = np.array(frame['transform_matrix'])[:3, :4]
                self.poses += [pose]
                c2w = torch.FloatTensor(pose)

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                self.image_paths += [image_path]
                img = Image.open(image_path)
                img = img.resize(self.img_wh, Image.LANCZOS)

                # for training, downscale h and w by X using simple averaging
                # get rid of extra rows and columns

                if self.opt.ds_method == 'lanc':
                    imgX = img.resize((self.img_wh[0]//self.opt.downscale, self.img_wh[1]//self.opt.downscale), Image.LANCZOS)
                    imgX = self.transform(imgX)
                    img = self.transform(img) # (3, h, w)
                elif self.opt.ds_method == 'avg':
                    img = self.transform(img)
                    imgX = F.avg_pool2d(img, self.opt.downscale)
                img = img.view(4, -1).permute(1, 0) # (h*w, 4) RGBA
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
                imgX = imgX.view(4, -1).permute(1, 0) # (h/X*w/X, 4) RGBA
                imgX = imgX[:, :3]*imgX[:, -1:] + (1-imgX[:, -1:]) # blend A to RGB

                if self.split == 'train_crop':
                    self.img_wh_ds = (self.img_wh[0] // self.opt.downscale, self.img_wh[1] // self.opt.downscale)
                    def center_crop(data, img_wh):
                        dH = int(img_wh[1]//2 * self.opt.precrop_frac)
                        dW = int(img_wh[0]//2 * self.opt.precrop_frac)
                        data = data.view(img_wh[1], img_wh[0], -1)
                        data = data[img_wh[1]//2 - dH: img_wh[1]//2 + dH, img_wh[0]//2 - dW: img_wh[0]//2 + dW, :]
                        data = data.reshape(2*dH * 2*dW, -1)
                        return data, dH, dW
                    imgX, _, _ = center_crop(imgX, self.img_wh_ds)

                self.all_rgbs += [imgX]
                if self.split == 'train_crop':
                    img, dH, dW = center_crop(img, self.img_wh)
                    img = img.view(self.img_wh[1] - 2*dH, self.img_wh[0] - 2*dW, -1)
                else:
                    img = img.view(self.img_wh[1], self.img_wh[0], -1)
                img = einops.rearrange(img, '(h s1) (w s2) c -> (h w) (s1 s2) c', 
                                        s1=self.opt.downscale, s2=self.opt.downscale) # (h/X*w/X, X*X, 8)
                self.all_rgbs_ori += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                if self.split == 'train_crop':
                    rays_o, dH, dW = center_crop(rays_o, self.img_wh)
                    rays_d, _, _ = center_crop(rays_d, self.img_wh)

                img_rays = torch.cat([rays_o, rays_d, 
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1) # (h*w, 8)
                if self.split == 'train_crop':
                    img_rays = img_rays.view(self.img_wh[1] - 2*dH, self.img_wh[0] - 2*dW, -1)
                else:
                    img_rays = img_rays.view(self.img_wh[1], self.img_wh[0], -1)
                img_rays = einops.rearrange(img_rays, '(h s1) (w s2) c -> (h w) (s1 s2) c', 
                                             s1=self.opt.downscale, s2=self.opt.downscale) # (h/X*w/X, X*X, 8)
                self.all_rays += [img_rays]

            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h/X*w/X, X*X, 8)
            self.all_rgbs_ori = torch.cat(self.all_rgbs_ori, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h/X*w/X, 3)

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
                      'rgbs_ori': self.all_rgbs_ori[idx],
                      'rgbs': self.all_rgbs[idx]}

        else: # create data for each image separately
            frame = self.meta['frames'][idx]
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))
            img = img.resize(self.img_wh, Image.LANCZOS)
            if self.opt.ds_method == 'lanc':
                imgX = img.resize((self.img_wh[0]//self.opt.downscale, self.img_wh[1]//self.opt.downscale), Image.LANCZOS)
                imgX = self.transform(imgX)
                img = self.transform(img)
            elif self.opt.ds_method == 'avg':
                img = self.transform(img) # (4, H, W)
                imgX = F.avg_pool2d(img, self.opt.downscale)
            else:
                raise Exception('Downscale option not found')

            valid_mask = (img[-1]>0).flatten() # (H*W) valid color area
            img = img.view(4, -1).permute(1, 0) # (H*W, 4) RGBA
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:]) # blend A to RGB
            img = img.view(self.img_wh[1], self.img_wh[0], -1)
            img = einops.rearrange(img, '(h s1) (w s2) c -> (h w) (s1 s2) c', 
                                    s1=self.opt.downscale, s2=self.opt.downscale)

            valid_maskX = (imgX[-1]>0).flatten()
            imgX = imgX.view(4, -1).permute(1, 0)
            imgX = imgX[:, :3]*imgX[:, -1:] + (1-imgX[:, -1:])

            rays_o, rays_d = get_rays(self.directions, c2w)

            rays = torch.cat([rays_o, rays_d,
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                              1) # (H*W, 8)
            raysX = rays.view(self.img_wh[0], self.img_wh[1], -1) 
            raysX = einops.rearrange(raysX, '(h s1) (w s2) c -> (h w) (s1 s2) c', 
                                     s1=self.opt.downscale, s2=self.opt.downscale) # (h/X*w/X, X*X, 8)

            sample = {'rays_ori': rays,
                      'rgbs_ori': img,
                      'c2w': c2w,
                      'valid_mask_ori': valid_mask,
                      'rays': raysX,
                      'rgbs': imgX,
                      'valid_mask': valid_maskX}

        return sample