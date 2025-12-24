# run_nerffacespeech.py 缓存机制修改文档

## 修改概述

本次修改为 `run_nerffacespeech.py` 添加了推理时缓存机制支持，通过缓存PTI（Pivotal Tuning Inversion）和3DMM拟合结果，显著加速同一角色的多次视频生成。

## 修改日期

2025.12.20

## 修改内容

### 1. 导入配置更新

**位置**：第8-16行

**修改前**：
```python
from config import (
    NERF_CONDA_ENV,
    NERF_CONDA_PYTHON,
    NERF_SCRIPT,
    NERF_CODE_DIR as NERF_WORKDIR,
    MODEL_DIR,
    get_character_test_image
)
```

**修改后**：
```python
from config import (
    NERF_CONDA_ENV,
    NERF_CONDA_PYTHON,
    NERF_SCRIPT,
    NERF_CODE_DIR as NERF_WORKDIR,
    MODEL_DIR,
    PROJECT_ROOT,  # 新增：用于构造缓存目录路径
    get_character_test_image
)

# 使用缓存版本的脚本（支持缓存机制）
NERF_SCRIPT_CACHE = NERF_WORKDIR / "StyleNeRF" / "main_NeRFFaceSpeech_audio_driven_w_given_poses_cache.py"
```

**说明**：
- 添加了 `PROJECT_ROOT` 导入，用于构造缓存目录的绝对路径
- 定义了 `NERF_SCRIPT_CACHE` 变量，指向支持缓存机制的脚本版本

### 2. 缓存目录构造逻辑

**位置**：第114-120行（在命令构造之前）

**新增代码**：
```python
# ---------- 构造缓存目录路径（PTI和3DMM缓存） ----------
# 缓存目录：项目根目录/assets/charactor/{角色名}
cache_dir = PROJECT_ROOT / "assets" / "charactor" / character
cache_dir = cache_dir.resolve()  # 转换为绝对路径
cache_dir.mkdir(parents=True, exist_ok=True)
add_log(f"[NeRF] 使用缓存目录: {cache_dir}", "info")
add_log(f"[NeRF] 缓存目录将存储PTI和3DMM拟合结果，可显著加速后续推理", "info")
```

**说明**：
- 缓存目录路径格式：`{项目根目录}/assets/charactor/{角色名}`
- 使用 `resolve()` 确保路径为绝对路径
- 自动创建缓存目录（如果不存在）
- 添加日志输出，便于调试和监控

### 3. 脚本路径选择逻辑

**位置**：第122-128行

**新增代码**：
```python
# 使用缓存版本的脚本
script_path = NERF_SCRIPT_CACHE if NERF_SCRIPT_CACHE.exists() else NERF_SCRIPT
if NERF_SCRIPT_CACHE.exists():
    add_log(f"[NeRF] 使用缓存版本脚本: {script_path}", "info")
else:
    add_log(f"[NeRF] 警告: 缓存版本脚本不存在，使用默认脚本: {script_path}", "warning")
    add_log(f"[NeRF] 缓存功能可能不可用", "warning")
```

**说明**：
- 优先使用缓存版本的脚本（`main_NeRFFaceSpeech_audio_driven_w_given_poses_cache.py`）
- 如果缓存版本不存在，回退到默认脚本
- 添加日志提示，便于排查问题

### 4. 命令参数更新

**位置**：第130-139行

**修改前**：
```python
cmd = [
    str(NERF_CONDA_PYTHON),
    str(NERF_SCRIPT),
    f"--outdir={output_path}",
    "--trunc=0.7",
    f"--network={network_path}",
    f"--test_data={audio_path}",
    f"--test_img={test_img}",
    "--motion_guide_img_folder=frames"
]
```

**修改后**：
```python
cmd = [
    str(NERF_CONDA_PYTHON),
    str(script_path),  # 使用动态选择的脚本路径
    f"--outdir={output_path}",
    "--trunc=0.7",
    f"--network={network_path}",
    f"--test_data={audio_path}",
    f"--test_img={test_img}",
    "--motion_guide_img_folder=frames",
    f"--cache_dir={cache_dir}"  # 新增：添加缓存目录参数
]
```

**说明**：
- 使用动态选择的 `script_path`（优先使用缓存版本）
- 添加 `--cache_dir` 参数，传入绝对路径的缓存目录

## 缓存机制工作原理

### 缓存目录结构

当使用缓存机制时，会在以下目录创建缓存文件：

```
{项目根目录}/assets/charactor/{角色名}/
├── G_PTI.pt                    # PTI微调后的生成器模型
├── w_PTI.pt                    # 潜在代码（latent codes）
├── bg_PTI.pt                   # 背景潜在代码
├── fitted_coeffs.pt            # 3DMM拟合系数
├── output_from_w_pivot_w_G_PTI.png  # 中间结果图像
├── img_tesnor_224_resize_prc.png    # 处理后的输入图像
└── fitted_img_*.png            # 3DMM拟合结果图像（如果save_option=True）
```

### 缓存流程

1. **首次运行**（无缓存）：
   - 执行PTI训练（耗时：几分钟到十几分钟）→ 保存 `G_PTI.pt`, `w_PTI.pt`, `bg_PTI.pt`
   - 执行3DMM拟合（耗时：几十秒到几分钟）→ 保存 `fitted_coeffs.pt`
   - 执行视频生成 → 保存最终视频

2. **后续运行**（有缓存）：
   - 检测到缓存文件存在 → 直接加载，跳过PTI训练和3DMM拟合
   - 直接执行视频生成（使用新的音频和动作引导）→ 保存最终视频

### 性能提升

- **首次运行**：10-20分钟（需要PTI训练和3DMM拟合）
- **后续运行**：几分钟（仅视频生成时间，跳过耗时步骤）
- **性能提升**：节省约70-90%的时间（取决于PTI和3DMM拟合的时间）

## 使用示例

### 代码调用方式

```python
from fastapi_server.utils.run_nerffacespeech import generate_video

# 首次生成（会创建缓存）
success = generate_video(
    audio_path="/path/to/audio1.wav",
    character="ayanami",
    output_path="/path/to/output1",
    model_name="stylegan2-ffhq-config-f.pkl"
)

# 第二次生成（使用缓存，速度更快）
success = generate_video(
    audio_path="/path/to/audio2.wav",
    character="ayanami",  # 相同角色
    output_path="/path/to/output2",
    model_name="stylegan2-ffhq-config-f.pkl"
)
```

### 缓存目录示例

假设项目根目录为 `/mnt/e/Documents/TFG_TALK_NeRFaceSpeech`，角色名为 `ayanami`：

```
/mnt/e/Documents/TFG_TALK_NeRFaceSpeech/assets/charactor/ayanami/
├── G_PTI.pt
├── w_PTI.pt
├── bg_PTI.pt
├── fitted_coeffs.pt
└── ...
```

## 注意事项

### 1. 缓存依赖输入图像

- PTI和3DMM缓存是针对特定输入图像生成的
- 如果更换角色的输入图像，应该清除旧缓存或使用不同的角色名

### 2. 磁盘空间

- 每个角色的缓存文件可能占用几百MB到几GB空间
- 建议定期清理不需要的缓存以节省磁盘空间

### 3. 缓存一致性

- 缓存文件是PyTorch的 `.pt` 格式
- 确保PyTorch版本兼容，否则可能无法加载

### 4. 多角色支持

- 不同角色使用不同的缓存目录，互不干扰
- 同一角色的多次推理可以共享缓存

## 兼容性说明

### 向后兼容

- 如果缓存版本脚本不存在，会自动回退到默认脚本
- 默认脚本不支持缓存功能，但不会报错
- 其他逻辑保持不变，不影响现有功能

### 脚本要求

- 需要 `main_NeRFFaceSpeech_audio_driven_w_given_poses_cache.py` 脚本存在
- 该脚本必须支持 `--cache_dir` 参数

## 测试建议

1. **首次运行测试**：
   - 使用新角色生成视频
   - 检查缓存目录是否正确创建
   - 验证缓存文件是否生成

2. **缓存加载测试**：
   - 使用相同角色再次生成视频
   - 检查日志中是否有"使用缓存目录"的提示
   - 验证是否跳过了PTI训练和3DMM拟合步骤

3. **性能对比测试**：
   - 记录首次运行时间
   - 记录后续运行时间
   - 验证性能提升是否符合预期

## 故障排查

### 问题1：缓存目录未创建

**可能原因**：
- 项目根目录路径不正确
- 权限不足

**解决方法**：
- 检查 `PROJECT_ROOT` 配置是否正确
- 检查目录权限

### 问题2：缓存未生效

**可能原因**：
- 缓存版本脚本不存在
- 脚本不支持 `--cache_dir` 参数

**解决方法**：
- 检查 `main_NeRFFaceSpeech_audio_driven_w_given_poses_cache.py` 是否存在
- 检查脚本是否支持 `--cache_dir` 参数

### 问题3：缓存文件损坏

**可能原因**：
- PyTorch版本不兼容
- 磁盘空间不足
- 文件系统错误

**解决方法**：
- 清除缓存目录，重新生成
- 检查PyTorch版本兼容性
- 检查磁盘空间和文件系统

## 后续优化建议

1. **缓存管理**：
   - 添加缓存清理功能
   - 添加缓存大小监控
   - 添加缓存有效性检查

2. **性能优化**：
   - 预加载常用角色的缓存
   - 使用SSD存储缓存目录
   - 批量处理时共享缓存

3. **功能增强**：
   - 支持缓存版本控制
   - 支持缓存压缩
   - 支持远程缓存存储

## 相关文件

- `fastapi_server/utils/run_nerffacespeech.py` - 修改的文件
- `NeRFFaceSpeech_Code/StyleNeRF/main_NeRFFaceSpeech_audio_driven_w_given_poses_cache.py` - 缓存版本脚本
- `NeRFFaceSpeech_Code/StyleNeRF/CACHE_MECHANISM.md` - 缓存机制详细说明文档
- `fastapi_server/config.py` - 配置文件（包含PROJECT_ROOT）

## 修改人员

（根据实际情况填写）

## 版本历史

- **v1.0** (2024-XX-XX): 初始版本，添加缓存机制支持

