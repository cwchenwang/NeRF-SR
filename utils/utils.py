"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import pickle
import importlib
from collections import defaultdict


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def one_hot(n, idx):
    x = torch.zeros(n, dtype=torch.float32)
    if n > 0:
        x[idx] = 1.
    return x


def batch_one_hot(n, idxs):
    return torch.stack([
        one_hot(n, idx) for idx in idxs
    ], dim=0)


def load_pickle(f):
    return pickle.load(open(f, 'rb'), encoding='latin1')


def save_pickle(obj, f):
    pickle.dump(obj, open(f, 'wb'))


def chunk_batch(func, chunk_size, *args, **kwargs):
    B = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            B = arg.shape[0]
            break
    out = defaultdict(list)
    out_dict = False
    for i in range(0, B, chunk_size):
        out_chunk = func(*[arg[i:i+chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args], **kwargs)
        if isinstance(out_chunk, torch.Tensor):
            out_chunk = {0: out_chunk}
            out_dict = False
        elif isinstance(out_chunk, dict):
            out_dict = True
        else:
            print(f'Return value of func must be in type [torch.Tensor, dict], get {type(out_chunk)}.')
            exit(1)
        for k, v in out_chunk.items():
            out[k].append(v)
    
    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    return out if out_dict else out[0]


def find_class_using_name(module_name, class_name, suffix='', type=object):
    """Import the module

    In the file, the class called ClassNameFuffix() will
    be instantiated. It has to be a subclass of type,
    and it is case-insensitive.
    """
    filename = module_name
    lib = importlib.import_module(filename)
    target = None
    target_class_name = class_name.replace('_', '') + suffix
    for name, cls in lib.__dict__.items():
        if name.lower() == target_class_name.lower() \
           and issubclass(cls, type):
            target = cls

    if target is None:
        print(f"In {filename}.py, there should be a subclass of {type} with class name that matches {target_class_name} in lowercase.")
        exit(0)

    return target
