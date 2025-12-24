# Requirements 拆分总结

## ✅ 拆分完成

已成功将 `nerffacespeech_requirements.txt` 拆分为以下文件：

### 📁 生成的文件

1. **`environment.min.yaml`** - 最小 conda 环境（只包含 Python 3.9 和 pip）
2. **`requirements.torch.txt`** - PyTorch 相关（3 个包）
   - torch==2.1.2+cu118
   - torchvision==0.16.2+cu118
   - torchaudio==2.1.2+cu118
3. **`requirements.pytorch3d.txt`** - PyTorch3D 相关（2 个包）
   - pytorch3d==0.7.8
   - nvdiffrast @ git+...
4. **`requirements.txt`** - 纯 Python 项目依赖（124 个包）

### 🗑️ 已删除的内容

- ❌ 所有 `@ file:///` 本地路径（5 个包，已替换为标准版本）
- ❌ 所有 `nvidia-*` CUDA runtime 包（13 个包）
- ❌ 所有 `mkl-*` conda 包（3 个包）

### ✅ 已修复的内容

以下包原本带有 `@ file:///` 路径，已替换为标准版本：
- colorama==0.4.6
- portalocker==2.10.0
- six==1.16.0
- tqdm==4.66.0

## 📊 统计信息

- **原始文件**: 145 个包
- **拆分后**: 124 (requirements.txt) + 3 (torch) + 2 (pytorch3d) = 129 个包
- **删除**: 20 个不兼容的包/路径

## 🚀 使用方法

详见 `INSTALL.md` 文件。

