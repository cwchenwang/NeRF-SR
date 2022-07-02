import torch
import torch.nn as nn
import torch.nn.functional as TF
import torchvision


class ColorMSELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss

class L1Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return loss

class PSNR(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
    
    def forward(self, inputs, targets, valid_mask=None):
        value = (inputs - targets)**2
        if valid_mask is not None:
            value = value[valid_mask]
        return -10 * torch.log10(torch.mean(value))

class GradLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).repeat(3, 1, 1).unsqueeze(0)
        kernel_y = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).repeat(3, 1, 1).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).to(opt.device)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).to(opt.device)

    def forward(self, inputs, targets):
        grad_x = TF.conv2d(inputs, self.weight_x)
        grad_y = TF.conv2d(inputs, self.weight_y)
        grad_inputs = grad_x ** 2 + grad_y ** 2
        grad_targets = TF.conv2d(targets, self.weight_x)**2 + TF.conv2d(targets, self.weight_x)**2
        return torch.mean(grad_inputs - grad_targets)**2

class TVLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
    
    def tensor_size(self, t):
        return t.shape[0] * t.shape[1] * t.shape[2]

    def forward(self, inputs):
        count_h = self.tensor_size(inputs[1:,:,:])
        count_w = self.tensor_size(inputs[:,1:,:])
        h_tv = torch.pow( (inputs[1:,:,:] - inputs[:-1,:,:]), 2 ).sum()
        w_tv = torch.pow( (inputs[:,1:,:] - inputs[:,:-1,:]), 2 ).sum()
        return h_tv / count_h + w_tv / count_w

class GradientLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def img_grad(self, x):
        # idea from tf.image.image_gradients(image)
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/image_ops_impl.py#L3441-L3512
        # x: (b,c,h,w), float32 or float64
        # dx, dy: (b,c,h,w)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        h_x = x.size()[-2]
        w_x = x.size()[-1]
        # gradient step=1
        left = x
        right = TF.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = TF.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
        # dx, dy = torch.abs(right - left), torch.abs(bottom - top)
        dx, dy = right - left, bottom - top 
        # dx will always have zeros in the last column, right-left
        # dy will always have zeros in the last row,    bottom-top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0
        return dx, dy
    
    def forward(self, inputs, targets):
        dx_inputs, dy_inputs = self.img_grad(inputs)
        dx_targets, dy_targets = self.img_grad(targets)
        return (TF.l1_loss(dx_inputs, dx_targets) + TF.l1_loss(dy_inputs, dy_targets))/2

class LaplacianLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.patch_size = opt.patch_size
        self.loss = lambda x: torch.mean(torch.abs(x))
    
    def forward(self, inputs):
        L1 = self.loss(inputs[:,:,:-2] + inputs[:,:,2:] - 2 * inputs[:,:,1:-1])
        L2 = self.loss(inputs[:,:-2,:] + inputs[:,2:,:] - 2 * inputs[:,1:-1,:])
        L3 = self.loss(inputs[:,:-2,:-2] + inputs[:,2:,2:] - 2 * inputs[:,1:-1,1:-1])
        L4 = self.loss(inputs[:,2:,:-2] + inputs[:,:-2,2:] - 2 * inputs[:,1:-1,1:-1])
        return (L1 + L2 + L3 + L4) / 4


class BilateralLaplacianLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.patch_size = opt.patch_size
        self.gamma = opt.bilateral_gamma
        self.loss = lambda x: torch.mean(torch.abs(x))
        self.bilateral = lambda x: torch.exp(-torch.abs(x).sum(-1) / self.gamma)
    
    def forward(self, inputs, weights):
        w1 = self.bilateral(weights[:,:,:-2] + weights[:,:,2:] - 2 * weights[:,:,1:-1])
        w2 = self.bilateral(weights[:,:-2,:] + weights[:,2:,:] - 2 * weights[:,1:-1,:])
        w3 = self.bilateral(weights[:,:-2,:-2] + weights[:,2:,2:] - 2 * weights[:,1:-1,1:-1])
        w4 = self.bilateral(weights[:,2:,:-2] + weights[:,:-2,2:] - 2 * weights[:,1:-1,1:-1])

        L1 = self.loss(w1 * (inputs[:,:,:-2] + inputs[:,:,2:] - 2 * inputs[:,:,1:-1]))
        L2 = self.loss(w2 * (inputs[:,:-2,:] + inputs[:,2:,:] - 2 * inputs[:,1:-1,:]))
        L3 = self.loss(w3 * (inputs[:,:-2,:-2] + inputs[:,2:,2:] - 2 * inputs[:,1:-1,1:-1]))
        L4 = self.loss(w4 * (inputs[:,2:,:-2] + inputs[:,:-2,2:] - 2 * inputs[:,1:-1,1:-1]))
        return (L1 + L2 + L3 + L4) / 4


class VGGPerceptualLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.input_scale = (-1, 1)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.ready = False

    def setup(self):
        self.vgg = torchvision.models.vgg19(pretrained=True).features

    def forward(self, es, ta):
        if not self.ready:
            self.setup()
            self.ready = True

        self.vgg = self.vgg.to(es.device)
        self.mean = self.mean.to(es.device)
        self.std = self.std.to(es.device)

        es = (es - self.input_scale[0]) / (self.input_scale[1] - self.input_scale[0])
        ta = (ta - self.input_scale[0]) / (self.input_scale[1] - self.input_scale[0])

        es = (es - self.mean) / self.std
        ta = (ta - self.mean) / self.std

        loss = []
        for midx, mod in enumerate(self.vgg):
            es = mod(es)
            with torch.no_grad():
                ta = mod(ta)
            if midx == 3:
                lam = 1
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 8:
                lam = 0.75
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 13:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 22:
                lam = 0.5
                loss.append(torch.abs(es - ta).mean() * lam)
            elif midx == 31:
                lam = 1
                loss.append(torch.abs(es - ta).mean() * lam)
                break
        return sum(loss)        


class SSIM():
    def __init__(self, data_range=(0, 1), kernel_size=(11, 11), sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian = gaussian
        
        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")
        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")
        
        data_scale = data_range[1] - data_range[0]
        self.c1 = (k1 * data_scale)**2
        self.c2 = (k2 * data_scale)**2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)
    
    def _uniform(self, kernel_size):
        max, min = 2.5, -2.5
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        for i, j in enumerate(kernel):
            if min <= j <= max:
                kernel[i] = 1 / (max - min)
            else:
                kernel[i] = 0

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size, sigma):
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        return torch.matmul(kernel_x.t(), kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    def __call__(self, output, target, reduction='mean'):
        if output.dtype != target.dtype:
            raise TypeError(
                f"Expected output and target to have the same data type. Got output: {output.dtype} and y: {target.dtype}."
            )

        if output.shape != target.shape:
            raise ValueError(
                f"Expected output and target to have the same shape. Got output: {output.shape} and y: {target.shape}."
            )

        if len(output.shape) != 4 or len(target.shape) != 4:
            raise ValueError(
                f"Expected output and target to have BxCxHxW shape. Got output: {output.shape} and y: {target.shape}."
            )

        assert reduction in ['mean', 'sum', 'none']

        channel = output.size(1)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1)

        output = TF.pad(output, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        target = TF.pad(target, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        input_list = torch.cat([output, target, output * output, target * target, output * target])
        outputs = TF.conv2d(input_list, self._kernel, groups=channel)

        output_list = [outputs[x * output.size(0) : (x + 1) * output.size(0)] for x in range(len(outputs))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        _ssim = torch.mean(ssim_idx, (1, 2, 3))

        if reduction == 'none':
            return _ssim
        return _ssim.mean() if reduction == 'mean' else _ssim.sum()
