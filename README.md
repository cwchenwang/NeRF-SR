# NeRF-SR: High-Quality Neural Radiance Fields using Supersampling

This is the official implementation of our ACM MM 2022 paper `NeRF-SR: High-Quality Neural Radiance Fields using Supersampling`. Pull requests and issues are welcome.

### [Project Page](https://cwchenwang.github.io/NeRF-SR) | [Video](https://youtu.be/c3Yx2nGvi8o) | [Paper](https://arxiv.org/abs/2112.01759)

Abstract: *We present NeRF-SR, a solution for high-resolution (HR) novel view synthesis with mostly low-resolution (LR) inputs. Our method is built upon Neural Radiance Fields (NeRF) that predicts per-point density and color with a multi-layer perceptron. While producing images at arbitrary scales, NeRF struggles with resolutions that go beyond observed images. Our key insight is that NeRF benefits from 3D consistency, which means an observed pixel absorbs information from nearby views. We first exploit it by a supersampling strategy that shoots multiple rays at each image pixel, which further enforces multi-view constraint at a sub-pixel level. Then, we show that NeRF-SR can further boost the performance of supersampling by a refinement network that leverages the estimated depth at hand to hallucinate details from related patches on an HR reference image. Experiment results demonstrate that NeRF-SR generates high-quality results for novel view synthesis at HR on both synthetic and real-world datasets.*

**Note: There is an error in the paper, for LLFF dataset training, the input resolution is 252x189, but the paper said it's 504x378.**

## Requirements
The codebase is tested on 
* Python 3.6.9 (should be compatible with Python 3.7+)
* PyTorch 1.8.1
* GeForce 1080Ti, 2080Ti, RTX 3090

Create a virtual environment and then run:
```
pip install -r requirements.txt
```

## Dataset
In our paper, we use the same dataset as in NeRF.
We only use LLFF dataset, but Blender is still possible:
- [LLFF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
- [Blender](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

However, our method is compatible to any dataset than can be trained on NeRF to perform super-resolution. Feel free to try out.

## Render the pretrained model
We provide pretrained models in [Google Drive](https://drive.google.com/drive/folders/1uLx2bbKzyJJMw3Nr3gOEo45acfpF0TUd?usp=sharing).

For supersampling, first download the pretrained models and put them under the `checkpoints/nerf-sr/${name}` directory, then run:
```bash
bash scripts/test_llff_downX.sh
```
or
```bash
bash scripts/test_blender_downX.sh
```
For the `${name}` parameter, you can directly use the one in the scripts. You can also modify it to your preference, then you have to change the script.

For refinement, run:
```bash
bash scripts/test_llff_refine.sh
```

## Train a new NeRF-SR model
Please check the configuration in the scripts. You can always modify it to your desired model config (especially the dataset path and input/output resolutions).
### Supersampling
```bash
bash scripts/train_llff_downX.sh
```
to train a 504x378 NeRF inputs.
or
```bash
bash scripts/train_blender_downX.sh
```

### Refinement
After supersampling and before refinement, we have to perform depth warping to find relevant patches, run:
```
python warp.py
```
to create `*.loc` files. An example of `*.loc` files can be found in the provided `fern` checkpoints (in the `30_val_vis` folder), which can be used directly for refinement.

After that, you can train the refinement model.
It is needed only for LLFF dataset:
```bash
bash scripts/train_llff_refine.sh
```


## Baseline Models
To replicate the results of baseline models, first train a vanilla NeRF using command:
```
bash scripts/train_llff.sh
```
or 
```
bash scrpts/train_blender.sh
```

For vanilla-NeRF, just test the trained NeRF under high resolutions using `bash scripts/test_llff.sh` or `bash scripts/test_blender.sh` (change the `img_wh` to your desired resolution). For NeRF-Bi, NeRF-Liif and NeRF-Swin, you need to super-resolve testing images with the corresponding model. The pretrained models of NeRF-Liif and NeRF-Swin can be found below:
- NeRF-Liif: We used the RDN-LIIF pretrained model. The download link can be found in the official [LIIF repo](https://github.com/yinboc/liif).
- NeRF-Swin: We used the "Real-world image SR" setting of [SwinIR](https://github.com/JingyunLiang/SwinIR) and the pretrained SwinIR-M model. Click to download the [x2](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x2_GAN.pth) and [x4](https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth) model.


## Citation
If you consider our paper or code useful, please cite our paper:
```
@inproceedings{wang2022nerf,
  title={NeRF-SR: High-Quality Neural Radiance Fields using Supersampling},
  author={Wang, Chen and Wu, Xian and Guo, Yuan-Chen and Zhang, Song-Hai and Tai, Yu-Wing and Hu, Shi-Min},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={6445--6454},
  year={2022}
}
```

## Credit
Our code borrows from [nerf_pl](https://github.com/kwea123/nerf_pl) and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
