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
from data.llff_dataset import \
    normalize, average_poses, center_poses, create_spiral_poses, create_spheric_poses
from torchvision.transforms import functional as TF
import cv2

class LLFFRefineDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--patch_len', type=int, default=64)
        parser.add_argument('--ref_idx', type=int, default=0)
        parser.add_argument('--syn_dataroot', type=str, required=True)
        parser.add_argument('--aug_num', type=int, default=200)
        parser.add_argument('--distort_scale', type=float, default=0.3)
        parser.add_argument('--num_ref_patches', type=int, default=8)
        parser.add_argument('--with_gt_patch', action='store_true')
        parser.add_argument('--ref_offset', type=int, default=64)
        parser.add_argument('--data_num', type=int, default=500000)
        parser.add_argument('--test_img_split', type=int, default=4)
        parser.set_defaults(white_bkgd=False, noise_std=1.)
        return parser
    
    def __init__(self, opt, mode):
        self.opt = opt
        self.mode = mode
        self.root_dir = opt.dataset_root
        self.split = mode
        assert self.split in ['train', 'val', 'test_train', 'test']
        self.img_wh = opt.img_wh
        # self.spheric_poses = opt.spheric_poses
        # self.val_num = max(1, opt.val_num)
        self.ref_idx = opt.ref_idx

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

        if self.split == 'train':
            image_path = self.image_paths[self.ref_idx]
            img = Image.open(image_path).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)

            img_sr = Image.open(os.path.join(self.opt.syn_dataroot, f'{self.ref_idx}-fine-ori.png')).convert('RGB')
            img_sr = img_sr.crop((0, 0, self.img_wh[0], self.img_wh[1]))
            assert img.size == img_sr.size, 'input size must be equal to synthesized image size'

            gt_pspc_imgs = [self.transform(img)]
            sr_pspc_imgs = [self.transform(img_sr)]
            bboxs = [torch.tensor([0, 0, self.img_wh[0], self.img_wh[1]])]
            for i in range(self.opt.aug_num - 1):
                startpoints, endpoints = T.RandomPerspective.get_params(self.img_wh[0], self.img_wh[1], self.opt.distort_scale)
                gt_pspc_img = TF.perspective(img, startpoints, endpoints)
                sr_pspc_img = TF.perspective(img_sr, startpoints, endpoints)
                gray_img = cv2.cvtColor(np.array(gt_pspc_img), cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contours[0])
                # print(x, y, x+w, y+h)
                bboxs.append(torch.tensor([x, y, x+w, y+h]))
                # fn_idx, b, c, s, h = \
                #     T.ColorJitter.get_params(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.05, 0.05))
                # gt_pspc_img = self.jitterImage(gt_pspc_img, fn_idx, b, c, s, h)
                # sr_pspc_img = self.jitterImage(sr_pspc_img, fn_idx, b, c, s, h)
                # gt_pspc_img.save(f'pspc/{i}-pspc_img.png')
                gt_pspc_imgs.append(self.transform(gt_pspc_img))
                sr_pspc_imgs.append(self.transform(sr_pspc_img))
            self.gt_pspc_imgs = torch.stack(gt_pspc_imgs, 0)
            self.sr_pspc_imgs = torch.stack(sr_pspc_imgs, 0) # (aug_num, h, w, 3)
            self.gt_img = self.transform(img)
            self.sr_img = self.transform(img_sr)
            self.bboxs = torch.stack(bboxs, 0)
        elif self.split == 'val' or self.split == 'test_train':
            gt_imgs = []
            sr_imgs = []
            for i, image_path in enumerate(self.image_paths):
                img = Image.open(image_path).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img_sr = Image.open(os.path.join(self.opt.syn_dataroot, f'{i}-fine-ori.png')).convert('RGB')
                img_sr = img_sr.crop((0, 0, self.img_wh[0], self.img_wh[1]))
                gt_imgs.append(self.transform(img))
                sr_imgs.append(self.transform(img_sr))
            self.sr_imgs = torch.stack(sr_imgs, 0)
            self.gt_imgs = torch.stack(gt_imgs, 0)
            self.ref_img = self.gt_imgs[self.ref_idx]
        elif self.split == 'test':
            sr_imgs = []
            for i in range(2): # hardcoded
                img_sr = Image.open(os.path.join(self.opt.syn_dataroot, f'{i}-fine-ori.png')).convert('RGB')
                img_sr = img_sr.crop((0, 0, self.img_wh[0], self.img_wh[1]))
                sr_imgs.append(self.transform(img_sr))
            self.sr_imgs = torch.stack(sr_imgs, 0)
            self.ref_img = self.transform(Image.open(self.image_paths[0]).convert('RGB').resize(self.img_wh, Image.LANCZOS))
            locs = []
            for i in range(2):
                loc = np.load(os.path.join(self.opt.syn_dataroot, '{}_locs.npz'.format(i)))['arr_0']
                locs.append(loc)
            self.locs = np.stack(locs, 0)
        if self.split == 'test_train':
            locs = []
            for i, _ in enumerate(self.image_paths):
                loc = np.load(os.path.join(self.opt.syn_dataroot, '{}_locs.npz'.format(i)))['arr_0']
                locs.append(loc)
            self.locs = np.stack(locs, 0)

    def jitterImage(self, img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = TF.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = TF.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = TF.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = TF.adjust_hue(img, hue_factor)
        return img
        
    def define_transforms(self):
        transform_list = [T.ToTensor()]
        transform_list += [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = T.Compose(transform_list)

    def get_start_pos(self, wl, hl, wh, hh):
        x_start = np.random.randint(wl, wh - self.opt.patch_len)
        y_start = np.random.randint(hl, hh - self.opt.patch_len)
        return x_start, y_start

    def __len__(self):
        if self.split == 'train':
            return self.opt.data_num
        elif self.split == 'val':
            return len(self.image_paths)
            # return self.opt.aug_num * (self.img_wh[0] - self.opt.patch_len + 1) * (self.img_wh[1] - self.opt.patch_len + 1)
        elif self.split == 'test_train':
            return len(self.image_paths) * self.opt.test_img_split
        elif self.split == 'test':
            return 120 * self.opt.test_img_split
            # return 1

    def __getitem__(self, idx):
        if self.split == 'train':
            img_idx = idx % self.opt.aug_num
            wl, hl, wh, hh = self.bboxs[img_idx]
            # if wl >= wh - self.opt.patch_len:
            #     print(wl, wh)
            # get positions
            x_start = np.random.randint(wl, wh - self.opt.patch_len)
            y_start = np.random.randint(hl, hh - self.opt.patch_len)
            # patch on sr image and gt patch
            sr_patch = self.sr_pspc_imgs[img_idx][:, y_start: y_start+self.opt.patch_len, x_start: x_start+self.opt.patch_len]
            gt_patch = self.gt_pspc_imgs[img_idx][:, y_start: y_start+self.opt.patch_len, x_start: x_start+self.opt.patch_len]
            # gt patch on reference image
            ref_wl = max(wl, x_start - self.opt.ref_offset)
            ref_wh = min(wh - self.opt.patch_len, x_start + self.opt.ref_offset)
            ref_hl = max(hl, y_start - self.opt.ref_offset)
            ref_hh = min(hh - self.opt.patch_len, y_start + self.opt.ref_offset)
            ref_patches = []
            for _ in range(self.opt.num_ref_patches):
                ref_x_start = np.random.randint(ref_wl, ref_wh)
                ref_y_start = np.random.randint(ref_hl, ref_hh)
                ref_patches.append(self.gt_img[:, ref_y_start: ref_y_start+self.opt.patch_len, ref_x_start: ref_x_start+self.opt.patch_len])
            if self.opt.with_gt_patch:
                ref_patches[np.random.randint(self.opt.num_ref_patches)] = gt_patch
            ref_patches = torch.stack(ref_patches, 0) # (num_ref_patches, 3, patch_len, patch_len)
        
        elif self.split == 'val':
            img_idx = idx % len(self.image_paths)
            x_start, y_start = self.get_start_pos(0, 0, self.img_wh[0], self.img_wh[1])
            sr_patch = self.sr_imgs[img_idx][:, y_start: y_start+self.opt.patch_len, x_start: x_start+self.opt.patch_len]
            gt_patch = self.gt_imgs[img_idx][:, y_start: y_start+self.opt.patch_len, x_start: x_start+self.opt.patch_len]
            # reference patch on reference image
            ref_patches = []
            ref_wl = max(0, x_start - self.opt.patch_len)
            ref_wh = min(self.img_wh[0] - self.opt.patch_len, x_start + self.opt.patch_len)
            ref_hl = max(0, y_start - self.opt.patch_len)
            ref_hh = min(self.img_wh[1] - self.opt.patch_len, y_start + self.opt.patch_len)
            for _ in range(self.opt.num_ref_patches):
                ref_x_start, ref_y_start = self.get_start_pos(ref_wl, ref_hl, ref_wh, ref_hh)
                ref_patches.append(self.ref_img[:, ref_y_start: ref_y_start+self.opt.patch_len, ref_x_start: ref_x_start+self.opt.patch_len])
            ref_patches = torch.stack(ref_patches, 0)
            # import time
            # T.ToPILImage()((torch.cat([sr_patch, gt_patch], dim=2)+1.0)/2.0).save(f'./test/{idx}-{time.time()}-gt-refine.png')
        elif self.split == 'test_train':
            sr_patch = []
            gt_patch = []
            ref_patches = []
            start_locs = []
            img_idx = idx // self.opt.test_img_split
            for i in range(0, self.img_wh[0], self.opt.patch_len):
                for j in range(0, self.img_wh[1], self.opt.patch_len):
                    x_start = min(self.img_wh[0] - self.opt.patch_len, i)
                    y_start = min(self.img_wh[1] - self.opt.patch_len, j)
                    # print(x_start, y_start)
                    start_locs.append(torch.Tensor([x_start, y_start]))
                    sr_patch.append(self.sr_imgs[img_idx][:, y_start: y_start+self.opt.patch_len, x_start: x_start+self.opt.patch_len])
                    gt_patch.append(self.gt_imgs[img_idx][:, y_start: y_start+self.opt.patch_len, x_start: x_start+self.opt.patch_len])
                    num_valid = 0
                    ref_patch = []
                    for m in range(x_start, x_start+self.opt.patch_len):
                        for n in range(y_start, y_start+self.opt.patch_len):
                            loc = self.locs[img_idx][n, m]
                            if 0 <= loc[0] and loc[0] < self.img_wh[0] \
                                and 0 <= loc[1] and loc[1] < self.img_wh[1]:
                                    ref_x_start = min(self.img_wh[0] - self.opt.patch_len, int(loc[0]))
                                    ref_y_start = min(self.img_wh[1] - self.opt.patch_len, int(loc[1]))
                                    ref_patch.append(self.ref_img[:, ref_y_start: ref_y_start+self.opt.patch_len, ref_x_start: ref_x_start+self.opt.patch_len])
                                    num_valid += 1
                                    if num_valid >= self.opt.num_ref_patches:
                                        break
                        if num_valid >= self.opt.num_ref_patches:
                            break
                    for _ in range(self.opt.num_ref_patches - len(ref_patch)):
                        ref_patch.append(self.sr_imgs[img_idx][:, y_start: y_start+self.opt.patch_len, x_start: x_start+self.opt.patch_len])
                    assert len(ref_patch) == self.opt.num_ref_patches
                    ref_patch = torch.stack(ref_patch, 0)
                    ref_patches.append(ref_patch)
            start_locs = torch.stack(start_locs, 0)
            ref_patches = torch.stack(ref_patches, 0) # (num_patches, num_ref_patches, C, 64, 64)
            sr_patch = torch.stack(sr_patch, 0) 
            gt_patch = torch.stack(gt_patch, 0) # (num_patches, C, 64, 64)
            chunk = idx % self.opt.test_img_split
            patches_per_chunk = sr_patch.shape[0] // self.opt.test_img_split
            sr_patch = sr_patch[chunk*patches_per_chunk: (chunk+1)*patches_per_chunk, :, :, :]
            gt_patch = gt_patch[chunk*patches_per_chunk: (chunk+1)*patches_per_chunk, :, :, :]
            ref_patches = ref_patches[chunk*patches_per_chunk: (chunk+1)*patches_per_chunk, :, :, :, :]
            start_locs = start_locs[chunk*patches_per_chunk: (chunk+1)*patches_per_chunk, :]
        elif self.split == 'test':
            sr_patch = []
            ref_patches = []
            start_locs = []
            img_idx = idx // self.opt.test_img_split
            for i in range(0, self.img_wh[0], self.opt.patch_len):
                for j in range(0, self.img_wh[1], self.opt.patch_len):
                    x_start = min(self.img_wh[0] - self.opt.patch_len, i)
                    y_start = min(self.img_wh[1] - self.opt.patch_len, j)
                    # print(x_start, y_start)
                    start_locs.append(torch.Tensor([x_start, y_start]))
                    sr_patch.append(self.sr_imgs[img_idx][:, y_start: y_start+self.opt.patch_len, x_start: x_start+self.opt.patch_len])
                    num_valid = 0
                    ref_patch = []
                    for m in range(x_start, x_start+self.opt.patch_len):
                        for n in range(y_start, y_start+self.opt.patch_len):
                            loc = self.locs[img_idx][n, m]
                            if 0 <= loc[0] and loc[0] < self.img_wh[0] \
                                and 0 <= loc[1] and loc[1] < self.img_wh[1]:
                                    ref_x_start = min(self.img_wh[0] - self.opt.patch_len, int(loc[0]))
                                    ref_y_start = min(self.img_wh[1] - self.opt.patch_len, int(loc[1]))
                                    ref_patch.append(self.ref_img[:, ref_y_start: ref_y_start+self.opt.patch_len, ref_x_start: ref_x_start+self.opt.patch_len])
                                    num_valid += 1
                                    if num_valid >= self.opt.num_ref_patches:
                                        break
                        if num_valid >= self.opt.num_ref_patches:
                            break
                    for _ in range(self.opt.num_ref_patches - len(ref_patch)):
                        ref_patch.append(self.sr_imgs[img_idx][:, y_start: y_start+self.opt.patch_len, x_start: x_start+self.opt.patch_len])
                    assert len(ref_patch) == self.opt.num_ref_patches
                    ref_patch = torch.stack(ref_patch, 0)
                    ref_patches.append(ref_patch)
            start_locs = torch.stack(start_locs, 0)
            ref_patches = torch.stack(ref_patches, 0) # (num_patches, num_ref_patches, C, 64, 64)
            sr_patch = torch.stack(sr_patch, 0) 
            chunk = idx % self.opt.test_img_split
            patches_per_chunk = sr_patch.shape[0] // self.opt.test_img_split
            sr_patch = sr_patch[chunk*patches_per_chunk: (chunk+1)*patches_per_chunk, :, :, :]
            ref_patches = ref_patches[chunk*patches_per_chunk: (chunk+1)*patches_per_chunk, :, :, :, :]
            start_locs = start_locs[chunk*patches_per_chunk: (chunk+1)*patches_per_chunk, :]

        sample = {
                'sr_patch': sr_patch,
                'ref_patches': ref_patches
            }
        if self.split != 'test':
            sample['gt_patch'] = gt_patch
        else:
            sample['gt_patch'] = torch.zeros_like(sample['sr_patch'])
        if self.split == 'test_train' or self.split == 'test':
            sample['start_locs'] = start_locs
            sample['wh'] = torch.Tensor(self.img_wh)
            sample['patch_len'] = self.opt.patch_len
        return sample
    
if __name__ == "__main__":
    dataset = LLFFRefineDataset()