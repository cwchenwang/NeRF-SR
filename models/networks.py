import math
from options import Configurable, str2bool
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn.parallel.distributed import DistributedDataParallel
from .embedding import BaseEmbedding
from utils.utils import find_class_using_name


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    if init_type != 'default':
        net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, opt):
    if opt.accelerator == 'dp':
        return init_net_dp(net, opt)
    elif opt.accelerator == 'ddp':
        return init_net_ddp(net, opt)


def init_net_dp(net, opt):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        device (str)       -- which device does the model move to

    Return an initialized network.
    """
    if opt.n_gpus > 0:
        assert(torch.cuda.is_available())
        net.to(opt.device)
        net = torch.nn.DataParallel(net)  # multi-GPUs
    init_weights(net, opt.init_type, init_gain=opt.init_gain)
    return net


def init_net_ddp(net, opt):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        device (str)       -- which device does the model move to

    Return an initialized network.
    """
    assert(torch.cuda.is_available())
    net.to(opt.device)
    net = DistributedDataParallel(net, device_ids=[opt.local_rank], output_device=opt.local_rank)
    init_weights(net, opt.init_type, init_gain=opt.init_gain)
    return net


def get_scheduler(optimizer, opt, last_epoch=-1):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs - opt.n_epochs_decay> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            t = max(0, epoch + 1 - opt.n_epochs + opt.n_epochs_decay) / float(opt.n_epochs_decay + 1)
            lr = opt.lr * (1 - t) + opt.lr_final * t
            return lr / opt.lr
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif opt.lr_policy == 'exp':
        def lambda_rule(epoch):
            t = max(0, epoch + 1 - opt.n_epochs + opt.n_epochs_decay) / float(opt.n_epochs_decay + 1)
            lr = math.exp(math.log(opt.lr) * (1 - t) + math.log(opt.lr_final) * t)
            return lr / opt.lr
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_epochs, gamma=opt.lr_decay_gamma, last_epoch=last_epoch)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler    


class VanillaMLP(nn.Module, Configurable):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--D', type=int, default=8)
        parser.add_argument('--W', type=int, default=256)
        parser.add_argument('--skips', type=int, nargs='+', default=[4])
        parser.add_argument('--stop_grad', type=str2bool, default=False)
        parser.add_argument('--no_dir', action='store_true')
        return parser

    def __init__(self, opt):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(VanillaMLP, self).__init__()
        self.opt = opt
        self.D = opt.D
        self.W = opt.W
        self.in_channels_xyz = find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_pos, opt.deg_pos, opt).out_channels
        self.in_channels_dir = find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_dir, opt.deg_dir, opt).out_channels
        self.skips = opt.skips
        self.out_channels_rgb = opt.dim_neutex if hasattr(opt, 'dim_neutex') else opt.dim_rgb

        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # direction encoding layers
        if not opt.no_dir:
            self.dir_encoding = nn.Sequential(
                                    nn.Linear(self.W + self.in_channels_dir, self.W//2),
                                    nn.ReLU(True))
        else:
            self.dir_encoding = nn.Sequential(
                nn.Linear(self.W, self.W//2),
                nn.ReLU(True)
            )

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        if opt.color_activation == 'sigmoid':
            color_activation = nn.Sigmoid()
        elif opt.color_activation == 'none':
            color_activation = nn.Identity()
        self.rgb = nn.Sequential(
            nn.Linear(self.W // 2, self.out_channels_rgb),
            color_activation
        )

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        input_xyz, input_dir = torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        if not self.opt.no_dir:
            dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        else:
            dir_encoding_input = xyz_encoding_final

        if self.opt.stop_grad:
            dir_encoding_input = dir_encoding_input.detach()

        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out


class NeuTexMLP(nn.Module, Configurable):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--D', type=int, default=8)
        parser.add_argument('--W', type=int, default=256)
        parser.add_argument('--skips', type=int, nargs='+', default=[4])
        return parser

    def __init__(self, opt):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(NeuTexMLP, self).__init__()
        self.opt = opt
        self.D = opt.D
        self.W = opt.W
        self.in_channels_xyz = find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_pos, opt.deg_pos, opt).out_channels
        self.in_channels_dir = find_class_using_name('models.embedding', opt.embedding, type=BaseEmbedding)(opt.dim_dir, opt.deg_dir, opt).out_channels
        self.skips = opt.skips
        self.out_channels_rgb = opt.dim_neutex if hasattr(opt, 'dim_neutex') else opt.dim_rgb

        # xyz encoding layers
        for i in range(self.D):
            if i == 0:
                layer = nn.Linear(self.in_channels_xyz, self.W)
            elif i in self.skips:
                layer = nn.Linear(self.W + self.in_channels_xyz, self.W)
            else:
                layer = nn.Linear(self.W, self.W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.W, self.W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(self.W + self.in_channels_dir, self.W),
                                nn.ReLU(True),
                                nn.Linear(self.W, self.W),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(self.W, 1)
        if opt.color_activation == 'sigmoid':
            color_activation = nn.Sigmoid()
        elif opt.color_activation == 'none':
            color_activation = nn.Identity()
        self.rgb = nn.Sequential(
            nn.Linear(self.W, self.out_channels_rgb),
            color_activation
        )

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        input_xyz, input_dir = torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)

        out = torch.cat([rgb, sigma], -1)

        return out

        
import functools
class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

class UnetGenerator(nn.Module, Configurable):
    """Create a Unet-based generator"""
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--learn_residual', action='store_true')
        parser.add_argument('--input_nc', type=int, default=27)
        parser.add_argument('--output_nc', type=int, default=3)
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--ngf', type=int, default=64)
        return parser

    def __init__(self, opt, num_downs=6, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()

        self.opt = opt
        output_nc = opt.output_nc
        input_nc = opt.input_nc
        ngf = opt.ngf
        norm_layer = get_norm_layer(self.opt.norm)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        if self.opt.learn_residual:
            return input[:, :3, :, :] + self.model(input)
        return self.model(input)

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--input_nc', type=int, default=27)
        parser.add_argument('--output_nc', type=int, default=3)
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        return parser

    def __init__(self, opt, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt

        output_nc = opt.output_nc
        input_nc = opt.input_nc

        norm_layer = get_norm_layer(self.opt.norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class NLayerDiscriminator(nn.Module, Configurable):
    """Defines a PatchGAN discriminator"""
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--input_nc', type=int, default=3)
        parser.add_argument('--ndf_dis', type=int, default=64)
        parser.add_argument('--n_layers_D', type=int, default=3)
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        return parser

    def __init__(self, opt):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        self.opt = opt
        assert(self.opt.patch_len == 32 or self.opt.patch_len == 64)

        ndf = opt.ndf_dis
        nc = opt.input_nc
        SN = torch.nn.utils.spectral_norm
        IN = lambda x : nn.InstanceNorm2d(x)

        blocks = []
        if self.opt.patch_len==64:
            blocks += [
                # input is (nc) x 64 x 64
                SN(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            blocks += [
                # input is (nc) x 32 x 32
                SN(nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False)),
                #nn.BatchNorm2d(ndf * 2),
                IN(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        blocks += [
            # state size. (ndf*2) x 16 x 16
            SN(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            #nn.BatchNorm2d(ndf * 4),
            IN(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SN(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            #nn.BatchNorm2d(ndf * 8),
            IN(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SN(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)),
            # nn.Sigmoid()
        ]
        blocks = [x for x in blocks if x]
        self.model = nn.Sequential(*blocks)

        # n_layers = opt.n_layers_D
        # norm_layer = get_norm_layer(opt.norm)
        # input_nc = opt.input_nc
        # ndf = opt.ndf

        # if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
        #     use_bias = norm_layer.func == nn.InstanceNorm2d
        # else:
        #     use_bias = norm_layer == nn.InstanceNorm2d

        # kw = 4
        # padw = 1
        # sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        # nf_mult = 1
        # nf_mult_prev = 1
        # for n in range(1, n_layers):  # gradually increase the number of filters
        #     nf_mult_prev = nf_mult
        #     nf_mult = min(2 ** n, 8)
        #     sequence += [
        #         nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
        #         norm_layer(ndf * nf_mult),
        #         nn.LeakyReLU(0.2, True)
        #     ]

        # nf_mult_prev = nf_mult
        # nf_mult = min(2 ** n_layers, 8)
        # sequence += [
        #     nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
        #     norm_layer(ndf * nf_mult),
        #     nn.LeakyReLU(0.2, True)
        # ]

        # sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        # self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='relu', normalization=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.normalization = {
            None: nn.Identity(),
            'batch': nn.BatchNorm2d(out_channels)
        }[normalization]
        self.activation = {
            'relu': nn.ReLU(inplace=True)
        }[activation]
    
    def forward(self, x):
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x


class Up2x(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', normalization=None):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')
        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0)
        self.normalization = {
            None: nn.Identity(),
            'batch': nn.BatchNorm2d(out_channels)
        }[normalization]
        self.activation = {
            'relu': nn.ReLU(inplace=True)
        }[activation]
    
    def forward(self, x):
        x = self.upsampling(x)
        x = self.pad(x)
        x = self.conv(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x

class Model_VNPCAT_Encoder(nn.Module):
    # Based on Unet and inpainting network
    def __init__(self, num_in_features = 3):
        super(Model_VNPCAT_Encoder, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(num_in_features, 128, 3, 1, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_bnorm = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_bnorm = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4_bnorm = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv5_bnorm = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv6_bnorm = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 512, 3, 2, 1)
        self.conv7_bnorm = nn.BatchNorm2d(512)

        self.apply(self.initialize_weight)

    def forward(self, x):

        # encoder
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2_bnorm(self.conv2(x1)))

        x3 = self.relu(self.conv3_bnorm(self.conv3(x2)))
        x4 = self.relu(self.conv4_bnorm(self.conv4(x3)))

        x5 = self.relu(self.conv5_bnorm(self.conv5(x4)))
        x6 = self.relu(self.conv6_bnorm(self.conv6(x5)))

        x7 = self.relu(self.conv7_bnorm(self.conv7(x6)))

        return [x2, x4, x6, x7]

    def initialize_weight(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


class Model_VNPCAT_Decoder(nn.Module):
    # Based on Unet and inpainting network
    def __init__(self):
        super(Model_VNPCAT_Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1 = nn.Conv2d(512*2, 512, 3, 1, 1)
        self.conv1_bnorm = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv2_bnorm = nn.BatchNorm2d(512)
        self.conv2_up = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv2_up_bnorm = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(512*3, 512, 3, 1, 1)
        self.conv3_bnorm = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_bnorm = nn.BatchNorm2d(512)
        self.conv4_up = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv4_up_bnorm = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256*3, 256, 3, 1, 1)
        self.conv5_bnorm = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv6_bnorm = nn.BatchNorm2d(256)
        self.conv6_up = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv6_up_bnorm = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128*3, 128, 3, 1, 1)
        self.conv7_bnorm = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv8_bnorm = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 3, 3, 1, 1)

        self.apply(self.initialize_weight)

    def forward(self, list_F_synth, list_F_max):

        # encoder
        F_synth_3 = list_F_synth[3]
        F_max_3 = list_F_max[3]
        x0 = torch.cat((F_synth_3, F_max_3), 1)
        x1 = self.relu(self.conv1_bnorm(self.conv1(x0)))
        x2 = self.relu(self.conv2_bnorm(self.conv2(x1)))
        x2_up = self.relu(self.conv2_up_bnorm(self.conv2_up(self.upsample(x2))))

        F_synth_2 = list_F_synth[2]
        F_max_2 = list_F_max[2]
        x2_cat = torch.cat((x2_up, F_synth_2, F_max_2), 1)
        x3 = self.relu(self.conv3_bnorm(self.conv3(x2_cat)))
        x4 = self.relu(self.conv4_bnorm(self.conv4(x3)))
        x4_up = self.relu(self.conv4_up_bnorm(self.conv4_up(self.upsample(x4))))

        F_synth_1 = list_F_synth[1]
        F_max_1 = list_F_max[1]
        x4_cat = torch.cat((x4_up, F_synth_1, F_max_1), 1)
        x5 = self.relu(self.conv5_bnorm(self.conv5(x4_cat)))
        x6 = self.relu(self.conv6_bnorm(self.conv6(x5)))
        x6_up = self.relu(self.conv6_up_bnorm(self.conv6_up(self.upsample(x6))))

        F_synth_0 = list_F_synth[0]
        F_max_0 = list_F_max[0]
        x6_cat = torch.cat((x6_up, F_synth_0, F_max_0), 1)
        x7 = self.relu(self.conv7_bnorm(self.conv7(x6_cat)))
        x8 = self.relu(self.conv8_bnorm(self.conv8(x7)))
        x9 = self.conv9(x8)
        x9 = self.tanh(x9)

        return x9

    def initialize_weight(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

class Model_VNPCAT_Decoder_NoPooling(nn.Module):
    # Based on Unet and inpainting network
    def __init__(self):
        super(Model_VNPCAT_Decoder_NoPooling, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv1_bnorm = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv2_bnorm = nn.BatchNorm2d(512)
        self.conv2_up = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv2_up_bnorm = nn.BatchNorm2d(512)

        self.conv3 = nn.Conv2d(512*2, 512, 3, 1, 1)
        self.conv3_bnorm = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_bnorm = nn.BatchNorm2d(512)
        self.conv4_up = nn.Conv2d(512, 256, 3, 1, 1)
        self.conv4_up_bnorm = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256*2, 256, 3, 1, 1)
        self.conv5_bnorm = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv6_bnorm = nn.BatchNorm2d(256)
        self.conv6_up = nn.Conv2d(256, 128, 3, 1, 1)
        self.conv6_up_bnorm = nn.BatchNorm2d(128)

        self.conv7 = nn.Conv2d(128*2, 128, 3, 1, 1)
        self.conv7_bnorm = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv8_bnorm = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 3, 3, 1, 1)

        self.apply(self.initialize_weight)

    def forward(self, list_F_synth):

        # encoder
        F_synth_3 = list_F_synth[3]
        # F_max_3 = list_F_max[3]
        # x0 = torch.cat((F_synth_3, F_max_3), 1)
        x0 = F_synth_3
        x1 = self.relu(self.conv1_bnorm(self.conv1(x0)))
        x2 = self.relu(self.conv2_bnorm(self.conv2(x1)))
        x2_up = self.relu(self.conv2_up_bnorm(self.conv2_up(self.upsample(x2))))

        F_synth_2 = list_F_synth[2]
        # F_max_2 = list_F_max[2]
        x2_cat = torch.cat((x2_up, F_synth_2), 1)
        x3 = self.relu(self.conv3_bnorm(self.conv3(x2_cat)))
        x4 = self.relu(self.conv4_bnorm(self.conv4(x3)))
        x4_up = self.relu(self.conv4_up_bnorm(self.conv4_up(self.upsample(x4))))

        F_synth_1 = list_F_synth[1]
        # F_max_1 = list_F_max[1]
        x4_cat = torch.cat((x4_up, F_synth_1), 1)
        x5 = self.relu(self.conv5_bnorm(self.conv5(x4_cat)))
        x6 = self.relu(self.conv6_bnorm(self.conv6(x5)))
        x6_up = self.relu(self.conv6_up_bnorm(self.conv6_up(self.upsample(x6))))

        F_synth_0 = list_F_synth[0]
        # F_max_0 = list_F_max[0]
        x6_cat = torch.cat((x6_up, F_synth_0), 1)
        x7 = self.relu(self.conv7_bnorm(self.conv7(x6_cat)))
        x8 = self.relu(self.conv8_bnorm(self.conv8(x7)))
        x9 = self.conv9(x8)
        x9 = self.tanh(x9)

        return x9

    def initialize_weight(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

class MaxPoolingModel(nn.Module, Configurable):
    # Based on Unet and inpainting network
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--not_use_ref', action='store_true')
        return parser

    def __init__(self, opt):
        super(MaxPoolingModel, self).__init__()
        self.opt = opt
        self.E = Model_VNPCAT_Encoder()
        if self.opt.not_use_ref:
            self.D = Model_VNPCAT_Decoder_NoPooling()
        else:
            self.D = Model_VNPCAT_Decoder()
        self.apply(self.initialize_weight)

    def forward(self, x_synth, list_x_candi):
        list_F_synth = self.E(x_synth)

        if self.opt.not_use_ref:
            x_refined = self.D(list_F_synth)
            return x_refined

        # reshape from (N, 8, C, H, W) to (N*8, C, H, W)
        candi_shape = list_x_candi.shape
        list_x_candi = list_x_candi.view(candi_shape[0]*candi_shape[1], candi_shape[2], candi_shape[3], candi_shape[4])
        list_list_F_candi = self.E(list_x_candi)
        concat_F0 = list_list_F_candi[0].view(candi_shape[0], candi_shape[1], list_list_F_candi[0].shape[-3], list_list_F_candi[0].shape[-2], list_list_F_candi[0].shape[-1])
        concat_F1 = list_list_F_candi[1].view(candi_shape[0], candi_shape[1], list_list_F_candi[1].shape[-3], list_list_F_candi[1].shape[-2], list_list_F_candi[1].shape[-1])
        concat_F2 = list_list_F_candi[2].view(candi_shape[0], candi_shape[1], list_list_F_candi[2].shape[-3], list_list_F_candi[2].shape[-2], list_list_F_candi[2].shape[-1])
        concat_F3 = list_list_F_candi[3].view(candi_shape[0], candi_shape[1], list_list_F_candi[3].shape[-3], list_list_F_candi[3].shape[-2], list_list_F_candi[3].shape[-1])

        F0_max, _ = torch.max(concat_F0, dim=1)
        F1_max, _ = torch.max(concat_F1, dim=1)
        F2_max, _ = torch.max(concat_F2, dim=1)
        F3_max, _ = torch.max(concat_F3, dim=1)

        list_F_max = [F0_max, F1_max, F2_max, F3_max]

        # # decoder
        x_refined = self.D(list_F_synth, list_F_max)

        return x_refined

    def initialize_weight(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.xavier_normal_(m.weight)

        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0) 