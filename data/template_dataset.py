import math
import torch
import numpy as np
from data.base_dataset import BaseDataset


class TemplateDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--input_dim', type=int, default=64)
        parser.add_argument('--output_dim', type=int, default=64)
        return parser

    def __init__(self, opt, mode):
        super().__init__(opt, mode)
        self.opt = opt
        self.mode = mode
        self.size = {
            'train': 10000,
            'val': 500,
            'test': 500
        }[mode]
        self.dim = opt.input_dim

    def __getitem__(self, i):
        x = torch.rand(self.dim)
        y = x**2
        return {
            'x': x,
            'y': y
        }
    
    def __len__(self):
        return self.size

