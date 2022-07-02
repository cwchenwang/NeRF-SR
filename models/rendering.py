import torch
import torch.nn as nn

class BilateralVolumetricRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if opt.sigma_activation == 'relu':
            self.sigma_activation = nn.ReLU(inplace=True)
        elif opt.sigma_activation == 'softplus':
            self.sigma_activation = lambda x: torch.log(1 + torch.exp(x - 1))
    
    def forward(self, rgb, neighbor_rgbs, sigma, z_vals, white_bkgd):
        """Volumetric Rendering Function.
        Args:
            rgb: color, [N_ray, N_samples, 3].
            sigma: density, [N_ray, N_samples].
            z_vals: depth, [N_ray, N_samples].
            white_bkgd: white background, bool.
        Returns:
            comp_rgb: jnp.ndarray(float32), [N_ray, 3].
            depth: jnp.ndarray(float32), [N_ray].
            opacity: jnp.ndarray(float32), [N_ray].
            weights: jnp.ndarray(float32), [N_ray, N_samples].
        """
        # Convert these values using volume rendering (Section 4)
        eps = 1e-10
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples)
        deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[:, :1])], -1)

        # if viewdir not normalized, do this:
        # dists = dists * jnp.linalg.norm(dirs[Ellipsis, None, :], axis=-1)

        # compute alpha by the formula (3)
        # the relu here is important!
        alpha = 1 - torch.exp(-deltas * self.sigma_activation(sigma)) # (N_rays, N_samples)
        accum_prod = torch.cat([
            torch.ones_like(alpha[:, :1]),
            torch.cumprod(1 - alpha[:, :-1] + eps, dim=-1)
        ], dim=-1)
        weights = alpha * accum_prod # (N_rays, N_samples)

        num_rays = rgb.shape[0]
        mask = weights.ge(1e-2)
        gamma = 1
        w1 = torch.exp(-torch.sum( (rgb[mask]-neighbor_rgbs[:num_rays][mask])**2, dim=-1 ) / gamma)
        w2 = torch.exp(-torch.sum( (rgb[mask]-neighbor_rgbs[num_rays:num_rays*2][mask])**2, dim=-1 ) / gamma)
        w3 = torch.exp(-torch.sum( (rgb[mask]-neighbor_rgbs[num_rays*2:num_rays*3][mask])**2, dim=-1 ) / gamma)
        w4 = torch.exp(-torch.sum( (rgb[mask]-neighbor_rgbs[num_rays*3:num_rays*4][mask])**2, dim=-1 ) / gamma)
        w5 = torch.exp(-torch.sum( (rgb[mask]-neighbor_rgbs[num_rays*4:num_rays*5][mask])**2, dim=-1 ) / gamma) 
        # import pdb; pdb.set_trace(); 
        rgb[mask] = rgb[mask] + neighbor_rgbs[:num_rays][mask]*w1.unsqueeze(-1) + neighbor_rgbs[num_rays:2*num_rays][mask]*w2.unsqueeze(-1) + neighbor_rgbs[2*num_rays:3*num_rays][mask]*w3.unsqueeze(-1) \
                         + neighbor_rgbs[3*num_rays:4*num_rays][mask]*w4.unsqueeze(-1) + neighbor_rgbs[4*num_rays:5*num_rays][mask]*w5.unsqueeze(-1)
        rgb[mask] = rgb[mask] / (1 + w1 + w2 + w3 + w4 + w5).unsqueeze(-1)

        comp_rgb = (weights[..., None] * rgb).sum(dim=-2) # (N_rays, 3)

        depth = (weights * z_vals).sum(dim=-1) # (N_rays)
        opacity = weights.sum(dim=-1) # (N_rays)

        if white_bkgd:
            comp_rgb += 1 - opacity[..., None]
        
        return comp_rgb, depth, opacity, weights

class VolumetricRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if opt.sigma_activation == 'relu':
            self.sigma_activation = nn.ReLU(inplace=True)
        elif opt.sigma_activation == 'softplus':
            self.sigma_activation = lambda x: torch.log(1 + torch.exp(x - 1))
    
    def forward(self, rgb, sigma, z_vals, white_bkgd):
        """Volumetric Rendering Function.
        Args:
            rgb: color, [N_ray, N_samples, 3].
            sigma: density, [N_ray, N_samples].
            z_vals: depth, [N_ray, N_samples].
            white_bkgd: white background, bool.
        Returns:
            comp_rgb: jnp.ndarray(float32), [N_ray, 3].
            depth: jnp.ndarray(float32), [N_ray].
            opacity: jnp.ndarray(float32), [N_ray].
            weights: jnp.ndarray(float32), [N_ray, N_samples].
        """
        # Convert these values using volume rendering (Section 4)
        eps = 1e-10
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples)
        deltas = torch.cat([deltas, 1e10 * torch.ones_like(deltas[:, :1])], -1)

        # if viewdir not normalized, do this:
        # dists = dists * jnp.linalg.norm(dirs[Ellipsis, None, :], axis=-1)

        # compute alpha by the formula (3)
        # the relu here is important!
        alpha = 1 - torch.exp(-deltas * self.sigma_activation(sigma)) # (N_rays, N_samples)
        accum_prod = torch.cat([
            torch.ones_like(alpha[:, :1]),
            torch.cumprod(1 - alpha[:, :-1] + eps, dim=-1)
        ], dim=-1)
        weights = alpha * accum_prod # (N_rays, N_samples)
        comp_rgb = (weights[..., None] * rgb).sum(dim=-2) # (N_rays, 3)
        depth = (weights * z_vals).sum(dim=-1) # (N_rays)
        opacity = weights.sum(dim=-1) # (N_rays)

        if white_bkgd:
            comp_rgb += 1 - opacity[..., None]
        
        return comp_rgb, depth, opacity, weights
