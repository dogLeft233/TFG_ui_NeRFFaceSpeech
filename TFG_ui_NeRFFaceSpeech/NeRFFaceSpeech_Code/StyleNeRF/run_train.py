#!/usr/bin/env python3
"""
StyleNeRF Training Script
支持从预训练模型开始微调训练
"""
import os
import sys
import click
import dnnlib
import legacy
from training.training_loop import training_loop
from training.dataset import ImageFolderDataset

#----------------------------------------------------------------------------

def parse_config_name(config_name: str) -> dict:
    """
    解析配置名称，返回对应的网络配置
    
    支持的配置:
    - style_ffhq_ae_basic: 基础 StyleNeRF FFHQ 配置
    """
    configs = {
        'style_ffhq_ae_basic': {
            'G_kwargs': {
                'class_name': 'training.stylenerf.Generator',
                'z_dim': 512,
                'c_dim': 0,
                'w_dim': 512,
                'img_resolution': 1024,
                'img_channels': 3,
                'mapping_kwargs': {
                    'num_layers': 8,
                },
                'synthesis_kwargs': {
                    'channel_base': 32768,
                    'channel_max': 512,
                    'num_fp16_res': 4,
                    'conv_clamp': 256,
                },
            },
            'D_kwargs': {
                'class_name': 'training.stylenerf.Discriminator',
                'c_dim': 0,
                'img_resolution': 1024,
                'img_channels': 3,
                'channel_base': 32768,
                'channel_max': 512,
                'num_fp16_res': 4,
                'conv_clamp': 256,
            },
            'loss_kwargs': {
                'class_name': 'training.loss.StyleGAN2Loss',
                'r1_gamma': 10.0,
                'style_mixing_prob': 0.9,
                'pl_weight': 0.0,
                'pl_batch_shrink': 2,
                'pl_decay': 0.01,
                'pl_no_weight_grad': False,
            },
            'augment_kwargs': {
                'class_name': 'training.augment.AugmentPipe',
                'xflip': 1,
                'rotate90': 0,
                'xint': 1,
                'scale': 1,
                'rotate': 0,
                'aniso': 0,
                'xfrac': 1,
                'brightness': 1,
                'contrast': 1,
                'lumaflip': 0,
                'hue': 1,
                'saturation': 1,
            },
        },
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]

#----------------------------------------------------------------------------

@click.command()
@click.option('--outdir', help='Where to save the training results', type=str, required=True, metavar='DIR')
@click.option('--data', 'data_path', help='Path to training dataset (directory or zip)', type=str, required=True, metavar='PATH')
@click.option('--resume', 'resume_pkl', help='Resume from network pickle', type=str, default=None, metavar='PKL')
@click.option('--resume-start', 'resume_start', help='Resume from step', type=int, default=0, metavar='INT')
@click.option('--model', 'model_config', help='Model configuration name', type=str, default='style_ffhq_ae_basic', metavar='STR')
@click.option('--kimg', 'total_kimg', help='Total training duration in kimg', type=int, default=50, metavar='INT')
@click.option('--batch', 'batch_size', help='Total batch size', type=int, default=4, metavar='INT')
@click.option('--batch-gpu', 'batch_gpu', help='Batch size per GPU', type=int, default=4, metavar='INT')
@click.option('--snap', 'network_snapshot_ticks', help='How often to save network snapshots', type=int, default=5, metavar='INT')
@click.option('--imgsnap', 'image_snapshot_ticks', help='How often to save image snapshots', type=int, default=1, metavar='INT')
@click.option('--aug', 'augment_p', help='Augmentation probability', type=str, default='noaug', metavar='STR')
@click.option('--mirror', 'xflip', help='Enable horizontal flip augmentation', is_flag=True, default=False)
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT')
@click.option('--gamma', 'r1_gamma', help='R1 regularization weight', type=float, default=10.0, metavar='FLOAT')
@click.option('--resolution', help='Image resolution', type=int, default=1024, metavar='INT')
@click.option('--seed', 'random_seed', help='Random seed', type=int, default=0, metavar='INT')
def main(
    outdir: str,
    data_path: str,
    resume_pkl: str,
    resume_start: int,
    model_config: str,
    total_kimg: int,
    batch_size: int,
    batch_gpu: int,
    network_snapshot_ticks: int,
    image_snapshot_ticks: int,
    augment_p: str,
    xflip: bool,
    gpus: int,
    r1_gamma: float,
    resolution: int,
    random_seed: int,
):
    """
    Train StyleNeRF model
    
    Example:
        python run_train.py \\
            --outdir=training_outputs/my_training \\
            --data=/path/to/training/images \\
            --resume=pretrained_networks/ffhq_1024.pkl \\
            --kimg=50 \\
            --batch=4 \\
            --snap=5
    """
    
    # 解析配置
    config = parse_config_name(model_config)
    
    # 设置数据增强概率
    if augment_p == 'noaug':
        augment_p_val = 0.0
    elif augment_p == 'ada':
        augment_p_val = 0.0  # 使用 ADA 自适应调整
        ada_target = 0.6
    else:
        try:
            augment_p_val = float(augment_p)
        except ValueError:
            raise click.BadParameter(f"Invalid augment_p value: {augment_p}. Use 'noaug', 'ada', or a float.")
    
    # 更新配置中的分辨率
    if resolution != 1024:
        config['G_kwargs']['img_resolution'] = resolution
        config['G_kwargs']['synthesis_kwargs']['img_resolution'] = resolution
        config['D_kwargs']['img_resolution'] = resolution
    
    # 更新 R1 gamma
    if r1_gamma != 10.0:
        config['loss_kwargs']['r1_gamma'] = r1_gamma
    
    # 准备训练集配置
    training_set_kwargs = dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDataset',
        path=data_path,
        resolution=resolution,
        use_labels=False,
        xflip=xflip,
    )
    
    # 准备数据加载器配置
    data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True,
        num_workers=3,
        prefetch_factor=2,
    )
    
    # 准备生成器配置
    G_kwargs = dnnlib.EasyDict(config['G_kwargs'])
    
    # 准备判别器配置
    D_kwargs = dnnlib.EasyDict(config['D_kwargs'])
    
    # 准备优化器配置
    G_opt_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam',
        lr=0.002,
        betas=[0.0, 0.99],
        eps=1e-8,
    )
    
    D_opt_kwargs = dnnlib.EasyDict(
        class_name='torch.optim.Adam',
        lr=0.002,
        betas=[0.0, 0.99],
        eps=1e-8,
    )
    
    # 准备数据增强配置
    if augment_p_val > 0 or augment_p == 'ada':
        augment_kwargs = dnnlib.EasyDict(config['augment_kwargs'])
    else:
        augment_kwargs = None
    
    # 准备损失函数配置
    loss_kwargs = dnnlib.EasyDict(config['loss_kwargs'])
    
    # 准备指标列表（可选）
    metrics = []
    
    # 设置 ADA 参数
    ada_target_val = None
    if augment_p == 'ada':
        ada_target_val = 0.6
    
    # 创建输出目录
    os.makedirs(outdir, exist_ok=True)
    
    # 打印配置信息
    print()
    print('Training configuration:')
    print(f'  Output directory: {outdir}')
    print(f'  Data path: {data_path}')
    print(f'  Resume from: {resume_pkl if resume_pkl else "None"}')
    print(f'  Resume start: {resume_start}')
    print(f'  Model config: {model_config}')
    print(f'  Total kimg: {total_kimg}')
    print(f'  Batch size: {batch_size} (per GPU: {batch_gpu})')
    print(f'  Resolution: {resolution}')
    print(f'  Augmentation: {augment_p} (p={augment_p_val})')
    print(f'  Mirror flip: {xflip}')
    print(f'  GPUs: {gpus}')
    print(f'  Random seed: {random_seed}')
    print()
    
    # 调用训练循环
    training_loop(
        run_dir=outdir,
        training_set_kwargs=training_set_kwargs,
        data_loader_kwargs=data_loader_kwargs,
        G_kwargs=G_kwargs,
        D_kwargs=D_kwargs,
        G_opt_kwargs=G_opt_kwargs,
        D_opt_kwargs=D_opt_kwargs,
        augment_kwargs=augment_kwargs,
        loss_kwargs=loss_kwargs,
        metrics=metrics,
        random_seed=random_seed,
        world_size=gpus,
        rank=0,
        gpu=0,
        batch_gpu=batch_gpu,
        batch_size=batch_size,
        ema_kimg=10.0,
        ema_rampup=0.05,
        G_reg_interval=4,
        D_reg_interval=16,
        augment_p=augment_p_val,
        ada_target=ada_target_val,
        ada_interval=4,
        ada_kimg=500,
        total_kimg=total_kimg,
        kimg_per_tick=4,
        image_snapshot_ticks=image_snapshot_ticks,
        network_snapshot_ticks=network_snapshot_ticks,
        resume_pkl=resume_pkl,
        resume_start=resume_start,
        cudnn_benchmark=True,
        allow_tf32=False,
        abort_fn=None,
        progress_fn=None,
        update_cam_prior_ticks=None,
        generation_with_image=False,
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

