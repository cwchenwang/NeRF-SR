import math
import torch
import numpy as np
from data.base_dataset import BaseDataset
import glob
import os
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T
from models.utils import *
from utils.colmap import \
    read_cameras_binary, read_images_binary, read_points3d_binary
import einops

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, pose_avg


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class LLFFDownXDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        parser.add_argument('--spheric_poses', action='store_true')
        parser.add_argument('--val_num', type=int, default=1)
        parser.add_argument('--unified_dir', action='store_true', help='use same view dir for all subpixels')
        parser.add_argument('--all_ref', action='store_true', help='use all images as ref for test')
        parser.add_argument('--include_var', action='store_true')
        parser.add_argument('--use_subset', action='store_true')
        parser.add_argument('--subset_num', type=int, default=20)
        parser.set_defaults(white_bkgd=False, noise_std=1.)
        return parser

    def __init__(self, opt, mode):
        self.opt = opt
        self.mode = mode

        self.root_dir = opt.dataset_root
        self.split = mode
        assert self.split in ['train', 'val', 'test', 'test_train', 'gan', 'reg_patch']
        self.img_wh = opt.img_wh
        self.spheric_poses = opt.spheric_poses
        self.val_num = max(1, opt.val_num) # at least 1
        
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

        if self.opt.sisr_path != None: # use SISR images for supervision
            print('Use SISR images')
            self.sr_image_paths = [os.path.join(self.opt.sisr_path, f) for f in sorted(os.listdir(self.opt.sisr_path)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
            if self.opt.use_subset:
                self.sr_image_paths = self.sr_image_paths[:self.opt.subset_num]

        if self.opt.use_subset:
            self.poses = self.poses[:self.opt.subset_num]
            self.image_paths = self.image_paths[:self.opt.subset_num]

        # ray directions for all pixels, same for all images (same H, W, focal)
        if not self.opt.unified_dir:
            self.directions = \
                get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal, self.opt.use_pixel_centers) # (H, W, 3)
        else:
            self.directions = get_ray_directions(self.img_wh[1]//self.opt.downscale, self.img_wh[0]//self.opt.downscale, \
                self.focal//self.opt.downscale, self.opt.use_pixel_centers) # (H/X, W/X, 3)
            self.directions = einops.repeat(self.directions, 'h w c -> (h s1) (w s2) c', s1=self.opt.downscale, s2=self.opt.downscale)
            
        if self.split == 'train' or self.split == 'gan' or self.split == 'reg_patch': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            self.all_rgbs_ori = []
            if self.opt.sisr_path != None:
                self.all_rgbs_sr = []
            self.reg_patch_len = self.opt.reg_patch_len
            self.n_img_patches = (self.img_wh[0]//self.opt.downscale - self.reg_patch_len + 1) * (self.img_wh[1]//self.opt.downscale - self.reg_patch_len + 1)
            self.n_patches = self.n_img_patches * (len(self.image_paths) - (0 if self.opt.include_var else 1))
            
            self.ref_idx = 0
            if self.opt.with_ref: 
                print('ref image is', self.image_paths[self.ref_idx])
                self.ref_rays = []
                self.ref_rgbs = []

            for i, image_path in enumerate(self.image_paths):
                if i == val_idx and not self.opt.include_var: # exclude the val image
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                if self.opt.sisr_path != None:
                    img_sr = Image.open(self.sr_image_paths[i]).convert('RGB')
                    assert img_sr.size[0] == self.img_wh[0] and img_sr.size[1] == self.img_wh[1], \
                        f'''sr image sizes mismatch'''
                    img_sr = self.transform(img_sr)
                    img_sr = img_sr.view(3, -1).permute(1, 0)
                    img_sr = img_sr.view(self.img_wh[1], self.img_wh[0], -1)
                    img_sr = einops.rearrange(img_sr, '(h s1) (w s2) c -> (h w) (s1 s2) c', 
                                        s1=self.opt.downscale, s2=self.opt.downscale) # (h/X*w/X, X*X, 3)
                    self.all_rgbs_sr += [img_sr]

                img = Image.open(image_path).convert('RGB')
                # assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                #     f'''{image_path} has different aspect ratio than img_wh, 
                #         please check your data!'''
                img = img.resize(self.img_wh, Image.LANCZOS)
                if self.opt.ds_method == 'lanc':
                    imgX = img.resize((self.img_wh[0]//self.opt.downscale, self.img_wh[1]//self.opt.downscale), Image.LANCZOS)
                    imgX = self.transform(imgX)
                    img = self.transform(img) # (3, h, w)
                elif self.opt.ds_method == 'avg':
                    img = self.transform(img)
                    imgX = F.avg_pool2d(img, self.opt.downscale)
                else:
                    raise Exception('Downscale option not found')

                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                imgX = imgX.view(3, -1).permute(1, 0)
                img = img.view(self.img_wh[1], self.img_wh[0], -1)
                img = einops.rearrange(img, '(h s1) (w s2) c -> (h w) (s1 s2) c', 
                                        s1=self.opt.downscale, s2=self.opt.downscale) # (h/X*w/X, X*X, 3)

                if self.opt.with_ref and i == self.ref_idx:
                    self.ref_rgbs = img
                self.all_rgbs += [imgX]
                self.all_rgbs_ori += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                                     # near plane is always at 1.0
                                     # near and far in NDC are always 0 and 1
                                     # See https://github.com/bmild/nerf/issues/34
                else:
                    # TODO: change this hard-coded factor
                    # See https://github.com/kwea123/nerf_pl/issues/50
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max()) # focus on central object only

                img_rays = torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1])],
                                             1) # (h*w, 8)
                img_rays = img_rays.view(self.img_wh[1], self.img_wh[0], -1)
                img_rays = einops.rearrange(img_rays, '(h s1) (w s2) c -> (h w) (s1 s2) c', 
                                             s1=self.opt.downscale, s2=self.opt.downscale) # (h/X*w/X, X*X, 8)
                if self.opt.with_ref and i == self.ref_idx:
                    self.ref_rays = img_rays
                self.all_rays += [img_rays]
                                 
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
            self.all_rgbs_ori = torch.cat(self.all_rgbs_ori, 0)
            if self.opt.sisr_path != None:
                self.all_rgbs_sr = torch.cat(self.all_rgbs_sr, 0)

            if self.opt.with_ref:
                self.ref_indices = torch.randperm(self.ref_rays.shape[0])

        elif self.split == 'val':
            print('val image is', self.image_paths[val_idx])
            self.val_idx = val_idx

        else: # for testing, create a parametric rendering path
            if self.split.endswith('train'): # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'reg_patch':
            return len(self.all_rays) // (self.opt.reg_patch_len**2)
        if self.split == 'val':
            return self.val_num
        if self.split == 'test_train':
            return len(self.poses)
        if self.split == 'gan':
            return (len(self.poses) - self.val_num) * \
                (self.img_wh[0]//self.opt.downscale-self.opt.patch_len//self.opt.downscale-1) * \
                (self.img_wh[1]//self.opt.downscale-self.opt.patch_len//self.opt.downscale-1)
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'rgbs_ori': self.all_rgbs_ori[idx]}
            if self.opt.sisr_path != None:
                sample['rgbs_sr'] = self.all_rgbs_sr[idx]
    
            # sample a patch from reference image
            if self.opt.with_ref and not self.opt.no_ref_loss:
                if idx > 0 and idx % self.ref_rays.shape[0] == 0:
                    self.ref_indices = torch.randperm(self.ref_rays.shape[0])
                # i_patch = torch.randint(self.ref_rays.shape[0], size=(1,))[0].item()
                i_patch = self.ref_indices[idx % self.ref_rays.shape[0]]
                sample['ref_rays'] = self.ref_rays[i_patch]
                sample['ref_rgbs'] = self.ref_rgbs[i_patch]
        elif self.split == 'reg_patch':
            # random a patch
            i_patch = torch.randint(high=self.n_patches, size=(1,))[0].item()
            i_img, i_patch = i_patch // self.n_img_patches, i_patch % self.n_img_patches
            row = i_patch // (self.img_wh[0]//self.opt.downscale - self.reg_patch_len + 1)
            col = i_patch % (self.img_wh[0]//self.opt.downscale - self.reg_patch_len + 1)
            start_idx = i_img * (self.img_wh[0]//self.opt.downscale) * (self.img_wh[1]//self.opt.downscale) + row*(self.img_wh[0]//self.opt.downscale) + col
            idxs = start_idx + torch.cat([torch.arange(self.reg_patch_len) + i*(self.img_wh[0]//self.opt.downscale) for i in range(self.reg_patch_len)])
            sample = {
                'patch_rays': self.all_rays[idxs],
                'patch_rgbs': self.all_rgbs[idxs],
                # 'rgbs_ori': self.all_rgbs_ori[idxs]
            }
            sample['patch_rays'] = einops.rearrange(sample['patch_rays'], '(h1 w1) (s1 s2) c -> (h1 s1) (w1 s2) c',
                                h1=self.opt.reg_patch_len, s1=self.opt.downscale)
            if self.opt.sisr_path != None:
               sample['rgbs_sr'] = self.all_rgbs_sr[idxs]

        elif self.split == 'gan':
            # random a patch from all possible locations
            def random_patch():
                start_loc = np.random.randint(len(self.poses) - self.val_num) \
                    * self.img_wh[0]//self.opt.downscale * self.img_wh[1]//self.opt.downscale
                i = np.random.randint(self.img_wh[1]//self.opt.downscale-self.opt.patch_len//self.opt.downscale)
                j = np.random.randint(self.img_wh[0]//self.opt.downscale-self.opt.patch_len//self.opt.downscale)
                row = torch.arange(i, i+self.opt.patch_len//self.opt.downscale) * self.img_wh[0]//self.opt.downscale
                col = torch.arange(j, j+self.opt.patch_len//self.opt.downscale)
                locs = row.view(-1,1).repeat(1, self.opt.patch_len//self.opt.downscale) + col.view(1, self.opt.patch_len//self.opt.downscale)
                #locs = row.repeat(self.opt.patch_len//self.opt.downscale, 1) + col.view(self.opt.patch_len//self.opt.downscale, 1)
                return locs, start_loc
            # locs, start_loc = random_patch()
            locsB, start_locB = random_patch()
            sample = {
                # 'gan_rays': self.all_rays[locs + start_loc],
                    #   'gan_rgbs': self.ref_rgbs[locs],
                      'gan_raysB': self.all_rays[locsB + start_locB],
                      'gan_rgbsB': self.ref_rgbs[locsB],
                    }
            if self.opt.all_ref:
                sample['gan_rgbsB'] = self.all_rgbs_ori[locsB + start_locB]
            for key in sample:
                sample[key] = einops.rearrange(sample[key], 'h1 w1 (s1 s2) c -> (h1 s1) (w1 s2) c',
                                  h1=self.opt.patch_len//self.opt.downscale, s1=self.opt.downscale)
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.poses[self.val_idx])
            elif self.split == 'test_train':
                c2w = torch.FloatTensor(self.poses[idx])
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                # TODO: change this hard-coded factor
                # See https://github.com/kwea123/nerf_pl/issues/50
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)
            raysX = rays.view(self.img_wh[1], self.img_wh[0], -1)
            raysX = einops.rearrange(raysX, '(h s1) (w s2) c -> (h w) (s1 s2) c', 
                                     s1=self.opt.downscale, s2=self.opt.downscale) # (h/X*w/X, X*X, 8

            sample = {'rays': raysX,
                      'c2w': c2w,
                      'rays_ori': rays}

            if self.split in ['val', 'test_train']:
                if self.split == 'val':
                    idx = self.val_idx
                img = Image.open(self.image_paths[idx]).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                imgX = F.avg_pool2d(img, self.opt.downscale)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                imgX = imgX.view(3, -1).permute(1, 0)
                img = img.view(self.img_wh[1], self.img_wh[0], -1)
                img = einops.rearrange(img, '(h s1) (w s2) c -> (h w) (s1 s2) c', 
                        s1=self.opt.downscale, s2=self.opt.downscale) # (h/X*w/X, X*X, 3)
                sample['rgbs'] = imgX
                sample['rgbs_ori'] = img

                if self.opt.sisr_path != None:
                    img_sr = Image.open(self.sr_image_paths[idx]).convert('RGB')
                    img_sr = self.transform(img_sr)
                    img_sr = img_sr.view(3, -1).permute(1, 0)
                    img_sr = img_sr.view(self.img_wh[1], self.img_wh[0], -1)
                    img_sr = einops.rearrange(img_sr, '(h s1) (w s2) c -> (h w) (s1 s2) c', 
                        s1=self.opt.downscale, s2=self.opt.downscale) # (h/X*w/X, X*X, 3)
                    sample['rgbs_sr'] = img_sr

        return sample
