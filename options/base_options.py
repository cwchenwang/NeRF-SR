import argparse
from models.base_model import BaseModel
from data.base_dataset import BaseDataset
import os
from utils import utils
import torch
import models
import data
import options
import json
from utils.utils import find_class_using_name


def add_dist_options(parser):
    """
    DDP is not fully tested:
    - may stuck at the end of training
    - potential performance degradation
    """
    parser.add_argument('--accelerator', type=str, default='dp', choices=['dp', 'ddp'])
    return parser


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser = add_dist_options(parser)

        parser.add_argument('--name', type=str, required=True, help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--summary_dir', type=str, default='./runs', help='models are saved here')
        parser.add_argument('--seed', type=int, default=99)

        # model parameters
        parser.add_argument('--model', type=str, default='template', help='chooses which model to use.')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [default | normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='template', help='chooses how datasets are loaded. [template]')
        parser.add_argument('--dataset_root', type=str, required=True)
        parser.add_argument('--sisr_path', type=str, default=None, help='path for SISR images')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=2048, help='input batch size')
        parser.add_argument('--eval_batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--img_wh', type=int, nargs=2, required=True)
        parser.add_argument('--use_pixel_centers', type=options.str2bool, default=True)
        parser.add_argument('--patch_size', type=int, default=1)

        # additional parameters
        parser.add_argument('--phase', type=str, choices=['train', 'test', 'infer'])
        parser.add_argument('--load_epoch', type=str, default='latest', help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # chunking
        parser.add_argument('--ray_chunk', type=int, default=4096)
        parser.add_argument('--point_chunk', type=int, default=2048 * 128)


        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = options.get_option_setter(find_class_using_name(f"models.{model_name}_model", model_name, 'model', BaseModel))
        parser = model_option_setter(parser)

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = options.get_option_setter(find_class_using_name(f"data.{dataset_name}_dataset", dataset_name, 'dataset', BaseDataset))
        parser = dataset_option_setter(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        opt_dict = {}
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
            opt_dict[k] = v
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        utils.mkdirs(expr_dir)
        with open(os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase)), 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        with open(os.path.join(expr_dir, '{}_opt.json'.format(opt.phase)), 'w') as opt_json_file:
            opt_json_file.write(json.dumps(opt_dict))

    def parse(self, rank):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.isTest = self.isTest
        opt.isInfer = self.isInfer

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # ddp
        if opt.accelerator == 'ddp':
            if not opt.isTrain:
                print('Distributed Data Parallel is only supported in training.')
                exit(0)
            opt.local_rank = rank

        opt.is_master = True if opt.accelerator == 'dp' else opt.local_rank == 0
        opt.n_gpus = torch.cuda.device_count()
        
        if opt.accelerator == 'dp':
            opt.device = 'cuda:0' if opt.n_gpus > 0 else 'cpu'
        elif opt.accelerator == 'ddp':
            opt.device = f"cuda:{opt.local_rank}"
        torch.cuda.set_device(opt.device)

        if opt.is_master:
            self.print_options(opt)

        self.opt = opt
        return self.opt
