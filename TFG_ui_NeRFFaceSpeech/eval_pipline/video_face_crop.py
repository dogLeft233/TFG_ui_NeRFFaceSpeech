"""对文件夹中的所有 mp4 文件进行人脸检测和裁剪。

核心功能：
- 使用 MTCNN 或 RetinaFace 检测人脸
- 采样检测 + 时间平滑，确保裁剪区域稳定
- 可设置人脸占画面的比重
- 保存处理后的 mp4 文件

用法示例：
    python scripts/video_face_crop.py \
        --input-dir data/geneface_datasets/data/raw/videos \
        --output-dir data/geneface_datasets/data/raw/videos_cropped \
        --face-ratio 0.6 \
        --detect-interval 30
        
"""

from __future__ import annotations

import argparse
import subprocess
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import scipy.ndimage

try:
    from facenet_pytorch import MTCNN  # type: ignore
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("[警告] 未安装 facenet_pytorch，将尝试使用 OpenCV 的 DNN 人脸检测")

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("[警告] 未安装 dlib，无法使用 FFHQ-style 裁剪")

# FFHQFaceAlignment
FFHQ_ALIGNMENT_AVAILABLE = False
try:
    import sys
    FFHQ_ALIGNMENT_DIR = Path(__file__).parent / "FFHQFaceAlignment"
    if FFHQ_ALIGNMENT_DIR.exists():
        sys.path.insert(0, str(FFHQ_ALIGNMENT_DIR))
        from lib.landmarks_pytorch import LandmarksEstimation
        import torch
        import PIL.Image
        import PIL.ImageFile
        import scipy.ndimage
        FFHQ_ALIGNMENT_AVAILABLE = True
except ImportError:
    print("[警告] 无法导入 FFHQFaceAlignment，请确保 FFHQFaceAlignment 目录存在且依赖已安装")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对视频进行人脸检测和裁剪")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="包含 mp4 文件的输入目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="保存处理后视频的输出目录",
    )
    parser.add_argument(
        "--face-ratio",
        type=float,
        default=0.6,
        help="人脸占画面的比例（0.0-1.0，默认0.6，即人脸占画面60%）",
    )
    parser.add_argument(
        "--detect-interval",
        type=int,
        default=30,
        help="每隔多少帧检测一次人脸（默认30，即每秒检测1次，假设25fps）",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(1024, 1024),
        help="输出视频尺寸（默认1024x1024）",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="裁剪中心平滑窗口大小（默认10帧）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若指定，则覆盖已存在的输出文件",
    )
    parser.add_argument(
        "--ffhq-style",
        action="store_true",
        help="使用 FFHQ-style 人脸对齐（需要 dlib 和 shape_predictor_68_face_landmarks.dat）",
    )
    parser.add_argument(
        "--landmark-model",
        type=Path,
        default=None,
        help="dlib 68点关键点模型路径（默认在 pretrained_networks 目录查找）",
    )
    parser.add_argument(
        "--resize-only",
        action="store_true",
        help="绕过人脸检测和裁剪，只对视频进行 resize（适用于已裁剪好的视频）",
    )
    parser.add_argument(
        "--ffhq-alignment",
        action="store_true",
        help="使用 FFHQFaceAlignment 进行对齐（从第一帧计算对齐参数，应用到所有帧）",
    )
    return parser.parse_args()


def detect_face_mtcnn(frame: np.ndarray, mtcnn: MTCNN) -> Optional[Tuple[int, int, int, int]]:
    """使用 MTCNN 检测人脸，返回 (x1, y1, x2, y2) 或 None。"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bboxes, probs = mtcnn.detect(rgb)
    if bboxes is not None and len(bboxes) > 0:
        # 选最大的人脸
        largest_idx = np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes])
        x1, y1, x2, y2 = bboxes[largest_idx].astype(int)
        return (x1, y1, x2, y2)
    return None


def detect_face_opencv(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """使用 OpenCV DNN 检测人脸（备用方案）。"""
    # 加载预训练的人脸检测模型
    model_path = Path(__file__).parent.parent / "NeRFFaceSpeech_Code" / "pretrained_networks"
    prototxt = model_path / "opencv_face_detector.pbtxt"
    model = model_path / "opencv_face_detector_uint8.pb"
    
    if not prototxt.exists() or not model.exists():
        return None
    
    net = cv2.dnn.readNetFromTensorflow(str(model), str(prototxt))
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
    net.setInput(blob)
    detections = net.forward()
    
    best_conf = 0
    best_bbox = None
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5 and confidence > best_conf:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            best_bbox = (x1, y1, x2, y2)
            best_conf = confidence
    
    return best_bbox


def get_landmarks_dlib(frame: np.ndarray, predictor: dlib.shape_predictor) -> Optional[np.ndarray]:
    """使用 dlib 获取 68 点人脸关键点。"""
    detector = dlib.get_frontal_face_detector()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = detector(rgb, 1)
    
    if len(dets) == 0:
        return None
    
    # 选择最大的人脸
    largest_idx = np.argmax([(d.right() - d.left()) * (d.bottom() - d.top()) for d in dets])
    d = dets[largest_idx]
    
    shape = predictor(rgb, d)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])
    return landmarks


def compute_ffhq_alignment_params(
    image: np.ndarray,
    landmarks: np.ndarray,
    transform_size: int = 1024,
) -> dict:
    """计算 FFHQ 对齐参数（从 align_crop_image 提取）。
    
    返回包含对齐参数的字典，可以用于后续帧的对齐。
    """
    lm = landmarks
    lm_eye_left = lm[36: 42]
    lm_eye_right = lm[42: 48]
    lm_mouth_outer = lm[48: 60]
    
    # Calculate auxiliary vectors
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    
    # Choose oriented crop rectangle
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    
    img = Image.fromarray(image)
    orig_size = img.size
    
    # Shrink
    shrink = int(np.floor(qsize / transform_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        quad = quad / shrink
        qsize = qsize / shrink
    else:
        rsize = orig_size
    
    # Crop
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, rsize[0]),
        min(crop[3] + border, rsize[1]),
    )
    
    # 计算 crop 后的 quad
    quad_after_crop = quad.copy()
    if crop[2] - crop[0] < rsize[0] or crop[3] - crop[1] < rsize[1]:
        quad_after_crop -= np.array([crop[0], crop[1]])
    
    # Pad
    pad = (
        int(np.floor(min(quad_after_crop[:, 0]))),
        int(np.floor(min(quad_after_crop[:, 1]))),
        int(np.ceil(max(quad_after_crop[:, 0]))),
        int(np.ceil(max(quad_after_crop[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - (crop[2] - crop[0]) + border, 0),
        max(pad[3] - (crop[3] - crop[1]) + border, 0),
    )
    
    # 最终的 quad（用于 transform）
    quad_final = quad_after_crop.copy()
    if max(pad) > border - 4:
        quad_final += np.array([pad[0], pad[1]])
    
    return {
        'shrink': shrink,
        'rsize': rsize,
        'crop': crop,
        'pad': pad,
        'quad': quad_final,
        'transform_size': transform_size,
        'border': border,
        'qsize': qsize,
    }


def apply_ffhq_alignment(
    frame: np.ndarray,
    alignment_params: dict,
) -> np.ndarray:
    """对单帧应用 FFHQ 对齐参数。"""
    shrink = alignment_params['shrink']
    rsize = alignment_params['rsize']
    crop = alignment_params['crop']
    pad = alignment_params['pad']
    quad = alignment_params['quad']
    transform_size = alignment_params['transform_size']
    border = alignment_params['border']
    
    img = Image.fromarray(frame)
    
    # Shrink
    if shrink > 1:
        img = img.resize(rsize, Image.Resampling.LANCZOS)
    
    # Crop
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
    
    # Pad
    enable_padding = True
    if enable_padding and max(pad) > border - 4:
        img_np = np.array(img, dtype=np.float32)
        img_np = np.pad(img_np, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img_np.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / (pad[0] + 1e-12), np.float32(w - 1 - x) / (pad[2] + 1e-12)),
            1.0 - np.minimum(np.float32(y) / (pad[1] + 1e-12), np.float32(h - 1 - y) / (pad[3] + 1e-12))
        )
        blur = alignment_params['qsize'] * 0.01
        img_np += (scipy.ndimage.gaussian_filter(img_np, [blur, blur, 0]) - img_np) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img_np += (np.median(img_np, axis=(0, 1)) - img_np) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img_np), 0, 255)), 'RGB')
    
    # Transform
    img = img.transform(
        (transform_size, transform_size),
        Image.Transform.QUAD,
        (quad + 0.5).flatten(),
        Image.Resampling.BILINEAR
    )
    
    return np.array(img)


def calculate_ffhq_crop_region(
    landmarks: np.ndarray,
    frame_h: int,
    frame_w: int,
    output_size: int,
) -> Tuple[int, int, int, int]:
    """根据 FFHQ 方法计算裁剪区域 (x1, y1, x2, y2)。"""
    # 提取关键点
    lm_eye_left = landmarks[36: 42]
    lm_eye_right = landmarks[42: 48]
    lm_mouth_outer = landmarks[48: 60]
    
    # 计算辅助向量
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    
    # 计算对齐矩形
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    
    # 计算裁剪区域
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))) - border,
        int(np.floor(min(quad[:, 1]))) - border,
        int(np.ceil(max(quad[:, 0]))) + border,
        int(np.ceil(max(quad[:, 1]))) + border,
    )
    
    # 确保裁剪区域在图像范围内
    crop = (
        max(crop[0], 0),
        max(crop[1], 0),
        min(crop[2], frame_w),
        min(crop[3], frame_h),
    )
    
    return crop


def calculate_crop_region(
    bbox: Tuple[int, int, int, int],
    frame_h: int,
    frame_w: int,
    face_ratio: float,
    output_w: int,
    output_h: int,
) -> Tuple[int, int, int, int]:
    """根据人脸框和 face_ratio 计算裁剪区域 (x1, y1, x2, y2)。"""
    x1, y1, x2, y2 = bbox
    face_w = x2 - x1
    face_h = y2 - y1
    face_center_x = (x1 + x2) // 2
    face_center_y = (y1 + y2) // 2
    
    # 根据 face_ratio 计算裁剪区域大小
    # face_ratio = 人脸大小 / 裁剪区域大小
    crop_size = max(face_w, face_h) / face_ratio
    
    # 确保裁剪区域不超过原图
    half_crop = int(crop_size / 2)
    crop_x1 = max(0, face_center_x - half_crop)
    crop_y1 = max(0, face_center_y - half_crop)
    crop_x2 = min(frame_w, face_center_x + half_crop)
    crop_y2 = min(frame_h, face_center_y + half_crop)
    
    return (crop_x1, crop_y1, crop_x2, crop_y2)


def smooth_crop_center(
    centers: deque[Tuple[int, int]], window_size: int
) -> Tuple[int, int]:
    """对裁剪中心进行滑动平均平滑。"""
    if len(centers) == 0:
        return (0, 0)
    
    recent = list(centers)[-window_size:]
    avg_x = int(np.mean([c[0] for c in recent]))
    avg_y = int(np.mean([c[1] for c in recent]))
    return (avg_x, avg_y)


def process_video(
    video_path: Path,
    output_path: Path,
    face_ratio: float,
    detect_interval: int,
    output_size: Tuple[int, int],
    smooth_window: int,
    overwrite: bool,
    ffhq_style: bool = False,
    landmark_model_path: Optional[Path] = None,
    resize_only: bool = False,
    ffhq_alignment: bool = False,
) -> None:
    """处理单个视频文件。"""
    if output_path.exists() and not overwrite:
        print(f"[跳过] 输出文件已存在: {output_path}")
        return
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # 默认帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[处理] {video_path.name}: {orig_w}x{orig_h}, {fps}fps, {total_frames}帧")
    
    # resize-only 模式：跳过所有人脸检测和裁剪
    if resize_only:
        print("[模式] resize-only: 跳过人脸检测和裁剪，直接 resize")
    elif ffhq_alignment:
        # FFHQFaceAlignment 模式
        if not FFHQ_ALIGNMENT_AVAILABLE:
            raise RuntimeError(
                "FFHQFaceAlignment 需要安装依赖。请确保：\n"
                "1. FFHQFaceAlignment 目录存在\n"
                "2. 已安装所需依赖（pip install -r FFHQFaceAlignment/requirements.txt）\n"
                "3. 已下载模型（python FFHQFaceAlignment/download.py）"
            )
        print("[模式] FFHQFaceAlignment: 从第一帧计算对齐参数，应用到所有帧")
        le = LandmarksEstimation(type='2D')
    else:
        # 初始化人脸检测器
        predictor = None
        if ffhq_style:
            if not DLIB_AVAILABLE:
                raise RuntimeError("FFHQ-style 裁剪需要 dlib 库，请安装: pip install dlib")
            
            # 查找关键点模型
            if landmark_model_path is None:
                model_path = Path(__file__).parent.parent / "NeRFFaceSpeech_Code" / "pretrained_networks"
                landmark_model_path = model_path / "shape_predictor_68_face_landmarks.dat"
            
            if not landmark_model_path.exists():
                raise FileNotFoundError(
                    f"未找到 dlib 关键点模型: {landmark_model_path}\n"
                    f"请从 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 下载并解压"
                )
            
            predictor = dlib.shape_predictor(str(landmark_model_path))
            print(f"[FFHQ] 使用关键点模型: {landmark_model_path}")
        else:
            if MTCNN_AVAILABLE:
                device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
                mtcnn = MTCNN(select_largest=True, device=device)
                detect_fn = lambda f: detect_face_mtcnn(f, mtcnn)
            else:
                detect_fn = detect_face_opencv
                if detect_fn(np.zeros((100, 100, 3), dtype=np.uint8)) is None:
                    print("[错误] 无法使用 OpenCV DNN 检测，请安装 facenet_pytorch")
                    cap.release()
                    return
    
    # 创建临时视频文件（只有视频，无音频）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_video = output_path.parent / f".temp_{output_path.name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_video), fourcc, fps, output_size)
    
    # 用于平滑的裁剪中心历史
    crop_centers: deque[Tuple[int, int]] = deque(maxlen=smooth_window * 2)
    current_crop_region: Optional[Tuple[int, int, int, int]] = None
    
    frame_idx = 0
    output_w, output_h = output_size
    
    # FFHQFaceAlignment: 在第一帧检测关键点并计算对齐参数
    alignment_params = None
    if ffhq_alignment:
        ret, first_frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.tensor(np.transpose(rgb_frame, (2, 0, 1))).float()
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            with torch.no_grad():
                landmarks, detected_faces = le.detect_landmarks(img_tensor.unsqueeze(0), detected_faces=None)
            
            if len(landmarks) > 0:
                landmarks_np = np.asarray(landmarks[0].detach().cpu().numpy())
                alignment_params = compute_ffhq_alignment_params(
                    rgb_frame,
                    landmarks_np,
                    transform_size=output_w
                )
                print(f"[FFHQAlignment] 在第一帧检测到关键点，对齐参数已计算")
            else:
                raise RuntimeError("FFHQFaceAlignment: 第一帧未检测到关键点")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到第一帧
    # resize-only 模式：跳过所有人脸检测逻辑
    elif resize_only:
        pass  # 不需要初始化检测器
    # FFHQ-style: 在第一帧检测关键点并计算对齐参数
    elif ffhq_style and predictor:
        ret, first_frame = cap.read()
        if ret:
            landmarks = get_landmarks_dlib(first_frame, predictor)
            if landmarks is not None:
                current_crop_region = calculate_ffhq_crop_region(
                    landmarks, orig_h, orig_w, output_w
                )
                print(f"[FFHQ] 在第一帧检测到关键点，裁剪区域: {current_crop_region}")
            else:
                print("[警告] FFHQ: 第一帧未检测到关键点，回退到普通检测")
                ffhq_style = False
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到第一帧
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # resize-only 模式：直接 resize，不进行任何裁剪
            if resize_only:
                resized = cv2.resize(frame, output_size, interpolation=cv2.INTER_CUBIC)
                out.write(resized)
                if (frame_idx + 1) % 100 == 0:
                    print(f"  已处理 {frame_idx + 1}/{total_frames} 帧 ({100*(frame_idx+1)/total_frames:.1f}%)")
                frame_idx += 1
                continue
            
            # FFHQFaceAlignment: 应用对齐参数
            if ffhq_alignment and alignment_params:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                aligned_frame = apply_ffhq_alignment(rgb_frame, alignment_params)
                # 转换为 BGR 并 resize 到输出尺寸（如果需要）
                if aligned_frame.shape[:2] != output_size[::-1]:
                    aligned_frame = cv2.resize(aligned_frame, output_size, interpolation=cv2.INTER_CUBIC)
                aligned_frame_bgr = cv2.cvtColor(aligned_frame, cv2.COLOR_RGB2BGR)
                out.write(aligned_frame_bgr)
                if (frame_idx + 1) % 100 == 0:
                    print(f"  已处理 {frame_idx + 1}/{total_frames} 帧 ({100*(frame_idx+1)/total_frames:.1f}%)")
                frame_idx += 1
                continue
            
            # FFHQ-style: 使用固定的裁剪区域（已在第一帧计算）
            if ffhq_style and current_crop_region:
                pass  # 使用已计算的裁剪区域
            # 普通模式: 每隔 detect_interval 帧检测一次
            elif not ffhq_style and frame_idx % detect_interval == 0:
                bbox = detect_fn(frame)
                if bbox:
                    crop_region = calculate_crop_region(
                        bbox, orig_h, orig_w, face_ratio, output_w, output_h
                    )
                    current_crop_region = crop_region
                    # 记录裁剪中心用于平滑
                    cx = (crop_region[0] + crop_region[2]) // 2
                    cy = (crop_region[1] + crop_region[3]) // 2
                    crop_centers.append((cx, cy))
                elif current_crop_region is None:
                    # 如果从未检测到人脸，使用整帧中心
                    crop_size = min(orig_w, orig_h)
                    cx, cy = orig_w // 2, orig_h // 2
                    current_crop_region = (
                        max(0, cx - crop_size // 2),
                        max(0, cy - crop_size // 2),
                        min(orig_w, cx + crop_size // 2),
                        min(orig_h, cy + crop_size // 2),
                    )
            
            # 如果有历史裁剪中心，进行平滑（仅普通模式）
            if not ffhq_style and len(crop_centers) > 1 and current_crop_region:
                smoothed_center = smooth_crop_center(crop_centers, smooth_window)
                x1, y1, x2, y2 = current_crop_region
                crop_w = x2 - x1
                crop_h = y2 - y1
                # 以平滑后的中心重新计算裁剪区域
                x1 = max(0, smoothed_center[0] - crop_w // 2)
                y1 = max(0, smoothed_center[1] - crop_h // 2)
                x2 = min(orig_w, x1 + crop_w)
                y2 = min(orig_h, y1 + crop_h)
                x1 = max(0, x2 - crop_w)
                y1 = max(0, y2 - crop_h)
                current_crop_region = (x1, y1, x2, y2)
            
            # 裁剪并调整大小
            if current_crop_region:
                x1, y1, x2, y2 = current_crop_region
                cropped = frame[y1:y2, x1:x2]
                resized = cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)
            else:
                # 如果没有裁剪区域，直接缩放整帧
                resized = cv2.resize(frame, output_size, interpolation=cv2.INTER_CUBIC)
            
            out.write(resized)
            
            if (frame_idx + 1) % 100 == 0:
                print(f"  已处理 {frame_idx + 1}/{total_frames} 帧 ({100*(frame_idx+1)/total_frames:.1f}%)")
            
            frame_idx += 1
    
    finally:
        cap.release()
        out.release()
    
    # 使用 ffmpeg 合并原始音频和裁剪后的视频
    print("[音频] 合并原始音频...")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(temp_video),
        "-i", str(video_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # 删除临时文件
        if temp_video.exists():
            temp_video.unlink()
        print(f"[完成] 已保存（含音频）: {output_path}")
    except subprocess.CalledProcessError as exc:
        # 如果合并失败，保留临时视频文件
        print(f"[警告] 音频合并失败，但视频已保存: {temp_video}")
        print(f"[警告] 错误: {exc}")
        if temp_video.exists():
            temp_video.rename(output_path)
            print(f"[完成] 已保存（无音频）: {output_path}")


def main() -> int:
    args = parse_args()
    
    if not args.input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {args.input_dir}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 mp4 文件
    videos = sorted(args.input_dir.glob("*.mp4")) + sorted(args.input_dir.glob("*.MP4"))
    if not videos:
        raise FileNotFoundError(f"在目录中未找到任何 mp4 文件: {args.input_dir}")
    
    print(f"[信息] 找到 {len(videos)} 个视频文件")
    print(f"[参数] 人脸占比: {args.face_ratio:.1%}, 检测间隔: {args.detect_interval}帧")
    print(f"[参数] 输出尺寸: {args.output_size[0]}x{args.output_size[1]}")
    
    for i, video_path in enumerate(videos):
        print("\n" + "=" * 60)
        print(f"[处理] ({i + 1}/{len(videos)}) {video_path.name}")
        print("=" * 60)
        
        output_path = args.output_dir / video_path.name
        
        try:
            process_video(
                video_path=video_path,
                output_path=output_path,
                face_ratio=args.face_ratio,
                detect_interval=args.detect_interval,
                output_size=tuple(args.output_size),
                smooth_window=args.smooth_window,
                overwrite=args.overwrite,
                ffhq_style=args.ffhq_style,
                landmark_model_path=args.landmark_model,
                resize_only=args.resize_only,
                ffhq_alignment=args.ffhq_alignment,
            )
        except Exception as exc:
            print(f"[错误] 处理视频 {video_path} 时出错: {exc}")
    
    print("\n" + "=" * 60)
    print("[完成] 全部视频处理结束")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

