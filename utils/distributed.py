import os
import torch
from torch.distributed import Backend

def setup_env(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    if opt.accelerator == 'dp':
        pass
    elif opt.accelerator == 'ddp':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12899'
        torch.distributed.init_process_group(
            backend=Backend.NCCL,
            init_method='env://',
            rank=opt.local_rank,
            world_size=opt.n_gpus
        )


def cleanup_env(opt):
    if opt.accelerator == 'dp':
        pass
    elif opt.accelerator == 'ddp':
        torch.distributed.destroy_process_group()