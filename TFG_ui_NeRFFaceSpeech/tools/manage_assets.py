import json
import os
from pathlib import Path
from typing import Dict, List


"""
Simple asset manager for model weights.

Usage (from repo root):

  # 1) 收集当前仓库中的模型权重到 assets/ 下，并用软链接替换原始位置
  python tools/manage_assets.py collect

  # 2) 在新环境中，从已经解压好的 assets/ 恢复各模型权重到原路径（以软链接形式）
  python tools/manage_assets.py restore

设计原则：
- 真实文件统一放在 assets/models/ 下
- 代码中使用的原始路径保持不变，但变为指向 assets 的软链接
- 所有映射关系记录在 assets/model_manifest.json 中，便于在新环境中恢复
"""


ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets"
MANIFEST_PATH = ASSETS_DIR / "model_manifest.json"

# 需要纳管的「模型权重 / 数据」根路径（相对仓库根目录）
# 单文件 + 目录混合；目录会递归收集常见权重后缀的文件
FILE_ROOTS: Dict[str, str] = {
    # 核心生成模型 & 分割模型 &辅助网络
    "NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl": "models",
    "NeRFFaceSpeech_Code/pretrained_networks/seg.pth": "models",
    "NeRFFaceSpeech_Code/pretrained_networks/LipaintNet.pt": "models",
    # SadTalker
    "NeRFFaceSpeech_Code/pretrained_networks/sad_talker_pretrained/SadTalker_V0.0.2_256.safetensors": "models/sadtalker",
    # dlib 人脸关键点
    "NeRFFaceSpeech_Code/pretrained_networks/shape_predictor_68_face_landmarks.dat": "models/dlib",
    # SyncNet / S3FD
    "metrics/scores_LSE/syncnet_python/data/syncnet_v2.model": "models/syncnet",
    "metrics/scores_LSE/syncnet_python/detectors/s3fd/weights/sfd_face.pth": "models/syncnet",
    # FFHQ 对齐（可选）
    "eval_pipline/FFHQFaceAlignment/lib/sfd/s3fd-619a316812.pth": "models/ffhq_align",
}

DIR_ROOTS: List[str] = [
    # BFM & Deep3DFaceRecon 目录通常包含多个 .mat / .pth / checkpoint 文件
    "NeRFFaceSpeech_Code/pretrained_networks/BFM",
    "NeRFFaceSpeech_Code/pretrained_networks/BFM_for_3DMM-Fitting-Pytorch",
    "NeRFFaceSpeech_Code/pretrained_networks/Deep3DFaceRecon_pytorch",
]

# 会被视为「权重/模型」的文件后缀
WEIGHT_SUFFIXES = (".pth", ".pt", ".pkl", ".safetensors", ".mat", ".model", ".dat")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT_DIR).as_posix()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect() -> None:
    """
    收集当前项目中的模型权重到 assets/ 下：
    - 将真实文件复制到 assets/models/... 中
    - 用软链接替换原始路径，指向 assets 中的真实文件
    - 记录 original -> asset 的映射到 assets/model_manifest.json
    """
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, str]] = []

    # 1) 处理单文件
    for rel_src, asset_root in FILE_ROOTS.items():
        src = ROOT_DIR / rel_src
        if not src.exists():
            # 对于可选文件（例如某些用户未下载），只提示，不报错
            print(f"[collect] SKIP (not found): {rel_src}")
            continue

        asset_base = ASSETS_DIR / asset_root
        _ensure_dir(asset_base)
        dst = asset_base / src.name

        # 复制到 assets
        if not dst.exists():
            _ensure_dir(dst.parent)
            print(f"[collect] COPY  {rel_src}  ->  {_rel(dst)}")
            dst.write_bytes(src.read_bytes())
        else:
            print(f"[collect] EXISTS in assets, skip copy: {_rel(dst)}")

        # 如果原路径还不是软链接，则替换为指向 assets 的软链接
        if src.is_symlink():
            print(f"[collect] SKIP link (already symlink): {rel_src}")
        else:
            src.unlink()
            rel_target = os.path.relpath(dst, start=src.parent)
            print(f"[collect] LINK  {rel_src}  ->  {rel_target}")
            os.symlink(rel_target, src)

        manifest.append({"original": rel_src, "asset": _rel(dst)})

    # 2) 递归处理目录（BFM / Deep3DFaceRecon 等）
    for rel_dir in DIR_ROOTS:
        src_dir = ROOT_DIR / rel_dir
        if not src_dir.exists():
            print(f"[collect] SKIP dir (not found): {rel_dir}")
            continue

        for path in src_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in WEIGHT_SUFFIXES:
                continue

            rel_src = _rel(path)
            # asset 路径：assets/models/<原始相对路径>
            dst = ASSETS_DIR / "models" / rel_src
            _ensure_dir(dst.parent)

            if not dst.exists():
                print(f"[collect] COPY  {rel_src}  ->  {_rel(dst)}")
                dst.write_bytes(path.read_bytes())
            else:
                print(f"[collect] EXISTS in assets, skip copy: {_rel(dst)}")

            if path.is_symlink():
                print(f"[collect] SKIP link (already symlink): {rel_src}")
            else:
                path.unlink()
                rel_target = os.path.relpath(dst, start=path.parent)
                print(f"[collect] LINK  {rel_src}  ->  {rel_target}")
                os.symlink(rel_target, path)

            manifest.append({"original": rel_src, "asset": _rel(dst)})

    # 3) 写出 manifest
    _ensure_dir(MANIFEST_PATH.parent)
    print(f"[collect] WRITE manifest: {_rel(MANIFEST_PATH)}")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("\n[collect] 完成。现在可以执行，例如：")
    print("  tar czf assets.tar.gz assets")


def restore() -> None:
    """
    从 assets/model_manifest.json 恢复各模型权重到原始路径（以软链接形式）。
    典型流程：在新环境中解压 assets/ 后执行。
    """
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"[restore] manifest not found: {_rel(MANIFEST_PATH)}")

    data = json.loads(MANIFEST_PATH.read_text())
    if not isinstance(data, list):
        raise SystemExit("[restore] manifest format error (expected list)")

    for entry in data:
        original_rel = entry.get("original")
        asset_rel = entry.get("asset")
        if not original_rel or not asset_rel:
            print(f"[restore] SKIP malformed entry: {entry}")
            continue

        src = ROOT_DIR / original_rel
        asset = ROOT_DIR / asset_rel

        if not asset.exists():
            print(f"[restore] MISSING asset, skip: {asset_rel}")
            continue

        _ensure_dir(src.parent)

        if src.exists() or src.is_symlink():
            # 如果已经是指向正确目标的软链接，则跳过
            if src.is_symlink():
                current_target = os.readlink(src)
                abs_current = (src.parent / current_target).resolve()
                if abs_current == asset.resolve():
                    print(f"[restore] OK (link exists): {original_rel}")
                    continue
            print(f"[restore] SKIP (exists, not overwritten): {original_rel}")
            continue

        rel_target = os.path.relpath(asset, start=src.parent)
        print(f"[restore] LINK  {original_rel}  ->  {rel_target}")
        os.symlink(rel_target, src)

    print("\n[restore] 完成。所有已在 manifest 中且存在于 assets/ 的权重，")
    print("         均已通过软链接恢复到原始路径。")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2 or sys.argv[1] not in {"collect", "restore"}:
        print("Usage:")
        print("  python tools/manage_assets.py collect   # 收集模型权重到 assets/ 并用软链接替换原始位置")
        print("  python tools/manage_assets.py restore   # 从 assets/ 恢复软链接到原始位置")
        raise SystemExit(1)

    cmd = sys.argv[1]
    if cmd == "collect":
        collect()
    else:
        restore()


