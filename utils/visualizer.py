import os
import sys
import time

import cv2
import torch
import numpy as np
from . import utils, html
import tempfile
from torch.utils.tensorboard import SummaryWriter
import imageio



def create_writer(opt):
    writer = SummaryWriter(os.path.join(opt.summary_dir, opt.name))
    return writer


class Visualizee():
    def __init__(self, type, data, tag='', subdir=None, timestamp=False, **kwargs):
        self.type = type
        self.data = data
        if isinstance(self.data, torch.Tensor):
            self.data = self.data.cpu().numpy()
        elif isinstance(self.data, list):
            self.data = [data.cpu().numpy() if isinstance(data, torch.Tensor) else data for data in self.data]
        self.tag = tag
        self.subdir = subdir
        self.timestamp = timestamp
        for k, v in kwargs.items():
            setattr(self, k, v)

def _save_pc(vis, global_step):
    pc = vis.data
    print(pc.shape)
    pc_name = f"{global_step}-{vis.name}.txt" if vis.timestamp and global_step is not None else f"{vis.name}.txt"
    np.savetxt(os.path.join(vis.save_dir, pc_name), pc)

def _save_image(vis, global_step):
    """
    type: 'image'
    tag: description in the filename, nullable
    name: identifier of the data
    data: 3xHxW / HxWx3; numpy.ndarray
    data_format: 'CHW' / 'HWC'
    range: (low, high)
    img_format: 'png'
    save_dir: saving directory
    """
    img = vis.data
    if vis.data_format == 'CHW':
        img = img.transpose(1, 2, 0)
    img = ((img - vis.range[0]) / (vis.range[1] - vis.range[0]) * 255.).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_name = f"{global_step}-{vis.name}.{vis.img_format}" if vis.timestamp and global_step is not None else f"{vis.name}.{vis.img_format}"
    cv2.imwrite(os.path.join(vis.save_dir, img_name), img)


def _save_gif(vis, global_step):
    """
    type: 'image'
    tag: description in the filename, nullable
    name: identifier of the data
    data: [3xHxW] / [HxWx3]; list of numpy.ndarray
    data_format: 'CHW' / 'HWC'
    range: (low, high)
    save_dir: saving directory
    """
    imgs = []
    for img in vis.data:
        if vis.data_format == 'CHW':
            img = img.transpose(1, 2, 0)
        img = ((img - vis.range[0]) / (vis.range[1] - vis.range[0]) * 255.).astype(np.uint8)
        imgs.append(img)
    out_name = f"{global_step}-{vis.name}.gif" if vis.timestamp and global_step is not None else f"{vis.name}.gif"
    imageio.mimsave(os.path.join(vis.save_dir, out_name), imgs, fps=30, palettesize=256)


def _save_images(vis, global_step):
    """
    type: 'images'
    tag: description in the filename, nullable
    names: [identifiers of the data]
    data: Nx3xHxW / NxHxWx3; numpy.ndarray
    data_format: 'NCHW' / 'NHWC'
    range: (low, high)
    img_format: 'png'
    subdir: relative to the save root, nullable    
    """    
    pass


def _save_matrix(vis, global_step):
    mat = vis.data
    cv2.imwrite(os.path.join(vis.save_dir, vis.name+'test-depth.png'), depth2im0(mat))
    mat = np.nan_to_num(mat) # change nan to 0
    mat_name = f"{global_step}-{vis.name}.npz" if vis.timestamp and global_step is not None else f"{vis.name}.npz"
    np.savez(os.path.join(vis.save_dir, mat_name), mat)

savers = {
    'image': _save_image,
    'images': _save_images,
    'gif': _save_gif,
    'matrix': _save_matrix,
    'pc': _save_pc
}


def _save_visual(vis, global_step):
    savers[vis.type](vis, global_step)

def save_visuals(save_dir, visuals, global_step=None):
    vis_all = []
    for name, vis in visuals.items():
        if isinstance(vis, Visualizee):
            vis_all.append(vis)
        elif isinstance(vis, list):
            vis_all += vis
    for vis in vis_all:
        assert vis.type in savers
        _save_dir = save_dir if vis.subdir is None else os.path.join(save_dir, vis.subdir)
        os.makedirs(_save_dir, exist_ok=True)
        setattr(vis, 'save_dir', _save_dir)

    for vis in vis_all:
        _save_visual(vis, global_step)


def depth2im(depth, cmap=cv2.COLORMAP_JET, size=None):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/max(ma-mi, 1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x = (cv2.applyColorMap(x, cmap) / 255.).astype(np.float32)
    if size is not None:
        x = cv2.resize(x, (size[1], size[0]))
    x = torch.from_numpy(x)
    return x

def depth2im0(x, cmap=cv2.COLORMAP_JET, size=None):
    """
    depth: (H, W)
    """
    # x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-0)/max(1-0, 1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x = (cv2.applyColorMap(x, cmap) / 255.).astype(np.float32)
    if size is not None:
        x = cv2.resize(x, (size[1], size[0]))
    x = (x * 255.).astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    # x = torch.from_numpy(x)
    return x
    
def depth2im(depth, near, far, cmap=cv2.COLORMAP_JET, size=None):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    x = (x-near)/max(far-near, 1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x = (cv2.applyColorMap(x, cmap) / 255.).astype(np.float32)
    if size is not None:
        x = cv2.resize(x, (size[1], size[0]))
    x = torch.from_numpy(x)
    return x