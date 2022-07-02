import torch
import numpy as np


def cast_rays(ori, dir, z_vals):
    """Stratified sampling along the rays.
    Args:
        ori: ray origins, [N_rays, 3].
        dir: ray directions, [N_rays, 3].
        z_vals: [N_rays, N_samples], sampled z values.
    Returns:
        points: [N_rays, N_samples, 3], sampled points.
    """        
    return ori[..., None, :] + z_vals[..., None] * dir[..., None, :]


def sample_along_rays(ori, dir, near, far, num_samples, randomized, lindisp):
    """Stratified sampling along the rays.
    Args:
        ori: ray origins, [N_rays, 3].
        dir: ray directions, [N_rays, 3].
        near: near clip, [N_rays].
        far: far clip, [N_rays].
        num_samples: int.
        randomized: bool, use randomized stratified sampling.
        lindisp: bool, sampling linearly in disparity rather than depth.
    Returns:
        z_vals: [N_rays, N_samples], sampled z values.
        points: [N_rays, N_samples, 3], sampled points.
    """    
    t_vals = torch.linspace(0, 1, num_samples, device=ori.device)
    if lindisp:
        z_vals = 1. / (1. / near * (1 - t_vals) + 1. / far * t_vals)
    else:
        z_vals = near * (1 - t_vals) + far * t_vals

    if randomized:
        z_mids = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
        upper = torch.cat([z_mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, :1], z_mids], dim=-1)
        z_vals = lower + torch.rand_like(z_vals) * (upper - lower)
    
    points = cast_rays(ori, dir, z_vals)
    return z_vals, points


def resample_along_rays(ori, dir, z_vals, weights, num_samples, randomized):
    """
    Sample @N_importance samples with distribution defined by @z_vals and @weights.
    Inputs:
        ori: ray origins, [N_rays, 3].
        dir: ray directions, [N_rays, 3].
        z_vals: [N_rays, N_samples], sampled z values.
        weights: [N_rays, N_samples]
        num_samples: int.
        randomized: bool, use randomized stratified sampling.
    Outputs:
        z_vals: [N_rays, N_samples], sampled z values.
        points: [N_rays, N_samples, 3], sampled points.
    """
    eps = 1e-5
    
    bins = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
    weights = weights[:, 1:-1]
    
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / weights.sum(dim=-1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive
    if randomized:
        u = torch.rand(N_rays, num_samples, device=ori.device)
    else:
        u = torch.linspace(0, 1, num_samples, device=ori.device)
        u = u.expand(N_rays, num_samples)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], dim=-1).view(N_rays, -1)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, -1, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, -1, 2)

    denom = cdf_g[...,1] - cdf_g[...,0]
    denom[denom < eps] = 1 # denom equals 0 means a bin has weight 0,
                         # in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    z_vals_ = bins_g[...,0] + (u - cdf_g[...,0]) / denom * (bins_g[...,1] - bins_g[...,0])
    z_vals = torch.sort(torch.cat([z_vals, z_vals_], dim=-1), dim=-1)[0]
    points = cast_rays(ori, dir, z_vals)
    return z_vals, points


def get_ray_directions(H, W, focal, use_pixel_centers=True):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length
        use_pixel_corners:
            If True, generate rays through the center of each pixel. Note: While
            this is the correct way to handle rays, it is not the way rays are
            handled in the original NeRF paper. Setting this TRUE yields ~ +1 PSNR
            compared to Vanilla NeRF.
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]
    
    # Projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return rays_o, rays_d


def add_gaussian_noise(raw, randomized, noise_std):
    """Adds gaussian noise to `raw`, which can used to regularize it.
    Args:
        key: jnp.ndarray(float32), [2,], random number generator.
        raw: jnp.ndarray(float32), arbitrary shape.
        noise_std: float, The standard deviation of the noise to be added.
        randomized: bool, add noise if randomized is True.
    Returns:
        raw + noise: jnp.ndarray(float32), with the same shape as `raw`.
    """
    if randomized and noise_std > 0:
        return raw + torch.randn_like(raw) * noise_std
    else:
        return raw
