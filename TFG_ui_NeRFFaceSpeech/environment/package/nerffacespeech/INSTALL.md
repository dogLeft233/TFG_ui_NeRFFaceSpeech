# NeRFFaceSpeech 环境安装指南

## 📋 安装顺序（严格按照此顺序执行）


### Step 1: 创建 conda 环境

```bash
conda env create -f environment.min.yaml
conda activate nerffacespeech
```

### Step 2: 安装 PyTorch（从官方源）

```bash
pip install -r requirements.torch.txt \
  --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: 安装 PyTorch3D 和 nvdiffrast

```bash
pip install -r requirements.pytorch3d.txt
```

### Step 4: 安装项目依赖

```bash
pip install -r requirements.txt
```

---

## 🐳 Docker 安装示例

```dockerfile
# 使用 conda 基础镜像
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# 复制环境文件
COPY environment.min.yaml requirements*.txt ./

# 创建 conda 环境
RUN conda env create -f environment.min.yaml -p /opt/env
ENV PATH=/opt/env/bin:$PATH

# 安装 PyTorch
RUN pip install -r requirements.torch.txt \
  --index-url https://download.pytorch.org/whl/cu118

# 安装 PyTorch3D
RUN pip install -r requirements.pytorch3d.txt

# 安装项目依赖
RUN pip install -r requirements.txt

# 复制项目代码
COPY . .

CMD ["python", "your_script.py"]
```

---

## ✅ 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch3d; print(f'PyTorch3D: {pytorch3d.__version__}')"
python -c "import nvdiffrast; print('nvdiffrast: OK')"
```

---

## 📝 文件说明

- `environment.min.yaml`: 最小 conda 环境（只包含 Python 和 pip）
- `requirements.torch.txt`: PyTorch 相关（torch, torchvision, torchaudio）
- `requirements.pytorch3d.txt`: PyTorch3D 和 nvdiffrast（需要单独安装）
- `requirements.txt`: 纯 Python 项目依赖（已排除 CUDA runtime、本地路径等）

---

## ⚠️ 注意事项

1. **不要使用 `pip freeze`** 直接导出，会包含 conda 本地路径和 CUDA runtime
2. **严格按照顺序安装**，PyTorch 必须在其他包之前安装
3. **nvdiffrast 需要编译**，确保系统有 CUDA toolkit 和编译工具
4. **mkl-* 包由 conda 管理**，不需要在 requirements.txt 中列出

