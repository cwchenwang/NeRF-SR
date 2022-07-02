import math
import torch
import numpy as np
from data.base_dataset import BaseDataset
import glob
import os
from PIL import Image
from torchvision import transforms as T
from models.utils import *
from utils.colmap import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from data.llff_dataset import \
    normalize, average_poses, center_poses, create_spiral_poses, create_spheric_poses
from tqdm import tqdm
import argparse

class LLFFDataset():

    def __init__(self, root_dir, result_dir, width, height):

        self.root_dir = root_dir
        self.result_dir = result_dir
        # self.root_dir = '/mnt/nas/raid/10102/nerf_data/nerf_llff_data/flower'
        # self.result_dir = '/home/wangchen/nerflex/checkpoints/llff-flower-378x504-ni64-dp/30_val_vis'
        self.split = 'train'
        self.img_wh = (width, height)
        self.spheric_poses = False
        
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        # Step 1: rescale focal length according to training resolution
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        H = camdata[1].height
        W = camdata[1].width
        self.focal = camdata[1].params[0] * self.img_wh[0]/W

        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        perm = np.argsort([imdata[k].name for k in imdata])
        # read successfully reconstructed images and ignore others
        self.image_paths = [os.path.join(self.root_dir, 'images', name)
                            for name in sorted([imdata[k].name for k in imdata])]
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4) cam2world matrices
        
        # read bounds
        self.bounds = np.zeros((len(poses), 2)) # (N_images, 2)
        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        pts_world = np.zeros((1, 3, len(pts3d))) # (1, 3, N_points)
        visibilities = np.zeros((len(poses), len(pts3d))) # (N_images, N_points)
        for i, k in enumerate(pts3d):
            pts_world[0, :, i] = pts3d[k].xyz
            for j in pts3d[k].image_ids:
                visibilities[j-1, i] = 1
        # calculate each point's depth w.r.t. each camera
        # it's the dot product of "points - camera center" and "camera frontal axis"
        depths = ((pts_world-poses[..., 3:4])*poses[..., 2:3]).sum(1) # (N_images, N_points)
        for i in range(len(poses)):
            visibility_i = visibilities[i]
            zs = depths[i][visibility_i==1]
            self.bounds[i] = [np.percentile(zs, 0.1), np.percentile(zs, 99.9)]
        # permute the matrices to increasing order
        poses = poses[perm]
        self.bounds = self.bounds[perm]
        
        # COLMAP poses has rotation in form "right down front", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        self.poses, _ = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
                                                   # center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        # TODO: change this hard-coded factor
        # See https://github.com/kwea123/nerf_pl/issues/50
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal, True) # (H, W, 3)
        
        nerf_depths = sorted(glob.glob(os.path.join(self.result_dir, '*fine-depth-ori.npz')))
        for i, image_path in enumerate(tqdm(self.image_paths)):
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            if i == 0:
                self.ref_rgbs = img
                self.ref_c2w = torch.FloatTensor(self.poses[i])
                self.ref_w2c = np.linalg.inv(np.concatenate([self.ref_c2w, np.array([0,0,0,1]).reshape(1,4)], 0))[:3]
            
            c2w = torch.FloatTensor(self.poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
            nerf_depth = np.load(os.path.join(self.result_dir, '{}-fine-depth-ori.npz'.format(i)))['arr_0']
            nerf_depth = np.squeeze(nerf_depth, axis=2)  # (h, w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                self.focal, 1.0, rays_o, rays_d) 
                nerf_depth = 1 / (1 - nerf_depth + 1e-6)
                                    # near plane is always at 1.0
                                    # near and far in NDC are always 0 and 1
                                    # See https://github.com/bmild/nerf/issues/34
            else:
                # TODO: change this hard-coded factor
                # See https://github.com/kwea123/nerf_pl/issues/50
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max()) # focus on central object only
            
            K = np.array([
                [self.focal, 0, 0.5*self.img_wh[0]],
                [0, -self.focal, 0.5*self.img_wh[1]],
                [0, 0, 1]
            ])
            # nerf_depth = -nerf_depth
            # K_inv = np.linalg.inv(K)
            i_idx, j_idx = np.meshgrid(
                np.arange(self.img_wh[0], dtype=np.float32) + 0.5,
                np.arange(self.img_wh[1], dtype=np.float32) + 0.5,
                indexing='xy'
            )
            # import pdb; pdb.set_trace()   
            coords = np.stack([(i_idx-self.img_wh[0]/2)/self.focal * nerf_depth, -(j_idx - self.img_wh[1]/2)/self.focal * nerf_depth, -nerf_depth], -1) # homogenous coordianate (h, w)
            coords_img = np.stack([(i_idx-self.img_wh[0]/2)/self.focal, -(j_idx - self.img_wh[1]/2)/self.focal, -np.ones_like(i_idx)], -1) # homogenous coordianate (h, w)
            # coords = np.stack([i_idx * nerf_depth, j_idx * nerf_depth, -nerf_depth], -1)
            
            warped_img = torch.zeros_like(img) # (h, w)
            # if i == 0:
            #     continue
            c2w = c2w.numpy()
            # if i != 15:
            #     continue
            # with open('15.txt', 'w') as f: # points in camera space
            #     for k in range(coords.shape[0]):
            #         for l in range(coords.shape[1]):
            #             f.write(f'{coords[k,l,0]} {coords[k,l,1]} {coords[k,l,2]} {img[0, k, l]} {img[1, k, l]} {img[2, k, l]}\n')
            #             f.write(f'{coords_img[k,l,0]} {coords_img[k,l,1]} {coords_img[k,l,2]} {img[0, k, l]} {img[1, k, l]} {img[2, k, l]}\n')
            for k in range(coords.shape[0]):
                for l in range(coords.shape[1]):
                    # if i >= 1:
                    # import pdb; pdb.set_trace();
                    # coords[k, l] = (c2w[:, :3] @ coords[k, l]) + c2w[:, 3] # world coordinate
                    coords[k, l] = c2w[:, :3] @ coords[k, l] + c2w[:, 3] # world coordinate
                    #f.write(f'{coords[k,l,0]} {coords[k,l,1]} {coords[k,l,2]} {img[0, k, l]} {img[1, k, l]} {img[2, k, l]}\n')

                    coords[k, l] = self.ref_w2c[:, :3] @ coords[k, l] + self.ref_w2c[:, 3]
                    coords[k, l] /= -coords[k,l,2]
                    # import pdb; pdb.set_trace()
                    coords[k,l,0] = int(coords[k,l,0] * self.focal + self.img_wh[0] / 2)
                    coords[k,l,1] = int(coords[k,l,1] * (-self.focal) + self.img_wh[1] / 2)
                    # import pdb; pdb.set_trace()

                    # coords[k,l,1] = self.img_wh[1] - coords[k,l,1]
                    if 0 <= coords[k,l,0] and coords[k,l,0] < self.img_wh[0] \
                        and 0 <= coords[k,l,1] and coords[k,l,1] < self.img_wh[1]:
                            warped_img[:, k, l] = self.ref_rgbs[:, int(coords[k,l,1]), int(coords[k,l,0])]
            T.ToPILImage()(warped_img).save(os.path.join(self.result_dir, f'{i}-wrapped.png'))
            np.savez(os.path.join(self.result_dir, f'{i}_locs.npz'), coords)
            

    def define_transforms(self):
        self.transform = T.ToTensor()

width = 504
height = 378

for scene in ['room']:
    print(scene)
    root_dir = f'/mnt/nas/raid/10102/nerf_data/nerf_llff_data/{scene}'
    result_dir = f'./checkpoints/nerf-sr/llff-{scene}-{height}x{width}-ni64-dp-ds2/30_val_vis'
    ds = LLFFDataset(root_dir, result_dir, width, height)