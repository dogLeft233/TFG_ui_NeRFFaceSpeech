import torch
import torch.nn as nn
from torchvision.models import inception_v3
from torchvision import transforms
from scipy import linalg
import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FIDCalculator:
    """FID计算器"""
    
    def __init__(self, device='cuda', batch_size=50):
        self.device = device
        self.batch_size = batch_size
        self.inception_model = None
        self._load_inception_model()
    
    def _load_inception_model(self):
        """加载InceptionV3模型"""
        logger.info("加载InceptionV3模型...")
        self.inception_model = inception_v3(pretrained=True, transform_input=False)
        # 移除最后的分类层，使用pool3特征
        self.inception_model.fc = nn.Identity()
        self.inception_model.eval()
        self.inception_model.to(self.device)
        logger.info("InceptionV3模型加载完成")
    
    def _preprocess_images(self, images):
        """预处理图像"""
        if isinstance(images, list):
            # 如果是路径列表，加载图像
            processed_images = []
            for img_path in images:
                if isinstance(img_path, str):
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = img_path
                
                # 调整大小到299x299
                img = cv2.resize(img, (299, 299))
                # 归一化到[0, 1]
                img = img.astype(np.float32) / 255.0
                # 转换为张量 [C, H, W]
                img = torch.from_numpy(img).permute(2, 0, 1)
                processed_images.append(img)
            
            images = torch.stack(processed_images)
        
        # 标准化（ImageNet统计量）
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225])
        images = normalize(images)
        
        return images.to(self.device)
    
    def extract_features(self, images):
        """
        提取图像特征
        Args:
            images: 图像张量 [N, 3, 299, 299] 或图像路径列表
        Returns:
            features: [N, 2048]
        """
        images = self._preprocess_images(images)
        
        all_features = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i+self.batch_size]
                features = self.inception_model(batch)
                all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    def calculate_statistics(self, features):
        """
        计算特征的均值和协方差矩阵
        Args:
            features: [N, 2048]
        Returns:
            mu: 均值向量 [2048]
            sigma: 协方差矩阵 [2048, 2048]
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        计算FID分数
        Args:
            mu1, sigma1: 真实图像的统计量
            mu2, sigma2: 生成图像的统计量
            eps: 数值稳定性参数
        Returns:
            fid: FID分数
        """
        # 计算均值差的平方和
        diff = mu1 - mu2
        mean_diff = np.sum(diff ** 2)
        
        # 计算协方差矩阵的矩阵平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 检查数值稳定性
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # 如果是复数，取实部
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # 计算FID
        trace_covmean = np.trace(covmean)
        fid = mean_diff + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_covmean
        
        return fid
    
    def compute_fid(self, real_images, generated_images):
        """
        计算FID分数
        Args:
            real_images: 真实图像路径列表或张量
            generated_images: 生成图像路径列表或张量
        Returns:
            fid_score: FID分数
        """
        logger.info("开始计算FID...")
        
        # 提取真实图像特征
        logger.info(f"提取真实图像特征，共{len(real_images)}张图像...")
        real_features = self.extract_features(real_images)
        
        # 提取生成图像特征
        logger.info(f"提取生成图像特征，共{len(generated_images)}张图像...")
        gen_features = self.extract_features(generated_images)
        
        # 计算统计量
        logger.info("计算统计量...")
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)
        
        # 计算FID
        fid_score = self.calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
        
        logger.info(f"FID计算完成: {fid_score:.2f}")
        return fid_score


def compute_fid(real_images, generated_images, device='cuda', batch_size=50):
    """
    计算FID分数的便捷函数
    Args:
        real_images: 真实图像路径列表或张量
        generated_images: 生成图像路径列表或张量
        device: 计算设备
        batch_size: 批处理大小
    Returns:
        fid_score: FID分数
    """
    calculator = FIDCalculator(device=device, batch_size=batch_size)
    return calculator.compute_fid(real_images, generated_images)


def extract_frames_from_video(video_path, max_frames=None):
    """从视频中提取所有帧或采样帧。
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大提取帧数，如果为 None 则提取所有帧
    
    Returns:
        帧列表，每个元素为 RGB 格式的 numpy 数组 [H, W, 3]，值域 [0, 255]
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames is not None and total_frames > max_frames:
        # 均匀采样
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        # 提取所有帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return frames


def compute_fid_from_videos(
    video1_path: Union[str, Path],
    video2_path: Union[str, Path],
    device: str = 'cuda',
    batch_size: int = 50,
    max_frames: Optional[int] = None
) -> float:
    """从两个视频文件计算 FID 分数。
    
    Args:
        video1_path: 第一个视频文件路径（MP4）
        video2_path: 第二个视频文件路径（MP4）
        device: 计算设备（'cuda' 或 'cpu'）
        batch_size: 批处理大小
        max_frames: 最大处理帧数，如果为 None 则处理所有帧
    
    Returns:
        FID 分数，值越小越好
    """
    # 提取帧
    frames1 = extract_frames_from_video(video1_path, max_frames)
    frames2 = extract_frames_from_video(video2_path, max_frames)
    
    # 对齐帧数（取较短的长度）
    min_frames = min(len(frames1), len(frames2))
    frames1 = frames1[:min_frames]
    frames2 = frames2[:min_frames]
    
    # 使用 FIDCalculator 计算 FID
    calculator = FIDCalculator(device=device, batch_size=batch_size)
    fid_score = calculator.compute_fid(frames1, frames2)
    
    return float(fid_score)


if __name__ == "__main__":
    # 测试代码
    print("FID计算器测试")
    
    # 创建一些测试数据
    test_images = [np.random.rand(299, 299, 3) for _ in range(10)]
    
    try:
        # 测试FID计算
        print("测试FID计算...")
        fid_calculator = FIDCalculator(device='cpu', batch_size=5)
        fid_score = fid_calculator.compute_fid(test_images, test_images)
        print(f"FID分数: {fid_score:.2f}")
            
    except Exception as e:
        print(f"测试失败: {str(e)}")

