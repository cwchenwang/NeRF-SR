"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def find_audio_preprocessor_using_name(ap_name):
    audiolib = importlib.import_module('data.audio')
    ap = None
    target_ap_name = ap_name.replace('_', '') + 'audiopreprocessor'
    for name, cls in audiolib.__dict__.items():
        if name.lower() == target_ap_name.lower():
            ap = cls

    if ap is None:
        print("In audio.py, there should be a class with class name that matches %s in lowercase." % (target_ap_name))
        exit(0)

    return ap 


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def get_audio_preprocessor_option_setter(ap_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    ap_class = find_audio_preprocessor_using_name(ap_name)
    return ap_class.modify_commandline_options    


def create_dataset(opt, mode, shuffle=True):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt, mode, shuffle)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, mode, shuffle):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.mode = mode
        if mode == 'train' or self.mode == 'train_crop':
            if opt.accelerator == 'ddp':
                assert opt.batch_size % opt.n_gpus == 0
                self.batch_size = opt.batch_size // opt.n_gpus
                if opt.is_master:
                    print(f'Using DDP, batch size per proc = {self.batch_size} ({opt.batch_size}/{opt.n_gpus})')
            else:
                self.batch_size = opt.batch_size
        else:
            self.batch_size = opt.eval_batch_size
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt, mode)
        if opt.is_master:
            print("dataset [%s] was created" % type(self.dataset).__name__)
        if opt.accelerator == 'dp' or self.mode != 'train':
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=int(opt.num_threads),
                drop_last=False if mode != 'train' else not opt.keep_last,
                pin_memory=True
            )
        else:
            sampler = torch.utils.data.DistributedSampler(
                self.dataset,
                num_replicas=opt.n_gpus,
                rank=opt.local_rank,
                shuffle=shuffle,
                seed=opt.seed
            )
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=int(opt.num_threads),
                drop_last=False if mode != 'train' else not opt.keep_last,
                sampler=sampler,
                pin_memory=True
            )

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.batch_size >= self.opt.max_dataset_size:
                break
            yield data
