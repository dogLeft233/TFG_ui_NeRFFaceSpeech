# ==================== 基础镜像 ====================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# ==================== 环境变量 ====================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    TORCH_HOME=/app/assets/.cache/torch \
    XDG_CACHE_HOME=/app/assets/.cache \
    HF_HOME=/app/assets/.cache/huggingface \
    CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_DIR=/etc/ssl/certs

# ==================== Ubuntu 镜像源 ====================
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list

# ==================== 系统依赖 ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget curl git build-essential cmake \
        ffmpeg libsm6 libxext6 libxrender-dev \
        libglib2.0-0 libgl1-mesa-glx \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ==================== 安装 Miniconda ====================
RUN wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda clean --all -y \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# ==================== Conda 镜像源 ====================
RUN /opt/conda/bin/conda config --remove-key channels || true \
    && /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
    && /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
    && /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ \
    && /opt/conda/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
    && /opt/conda/bin/conda config --set channel_priority strict \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ==================== 工作目录 ====================
WORKDIR /app

# ==================== 只复制环境定义（缓存关键点） ====================
COPY environment/package/ /app/environment/package/
COPY environment/*.yaml /app/environment/

# ==================== 创建目录 ====================
RUN mkdir -p \
    /app/assets/.cache/{torch,huggingface} \
    /app/assets/models \
    /app/database/videos \
    /app/outputs/{audio,video} \
    /app/NeRFFaceSpeech_Code/pretrained_networks \
    /app/environment/{nerffacespeech,api,syncnet,llm_talk}

# ==================== 声明挂载卷（与 Dockerfile_ 对齐，提升缓存可复用） ====================
VOLUME ["/app/assets/.cache", "/app/assets/models", "/app/outputs", "/app/database"]

# =====================================================================
# ==================== nerffacespeech 环境 =============================
# =====================================================================

# 1. 只创建 conda 环境（最稳定层）
RUN conda env create \
    -f /app/environment/package/nerffacespeech/environment.min.yaml \
    -p /app/environment/nerffacespeech

RUN conda env create \
    -f /app/environment/package/api/environment.min.yaml \
    -p /app/environment/api

RUN conda env create \
    -f /app/environment/package/syncnet/environment.min.yaml \
    -p /app/environment/syncnet

RUN conda env create \
    -f /app/environment/package/llm_talk/environment.min.yaml \
    -p /app/environment/llm_talk

# =====================================================================
# ==================== api 环境 =======================================
# =====================================================================

# =====================================================================
# ==================== syncnet 环境 ===================================
# =====================================================================

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh \
    && conda activate /app/environment/syncnet \
    && pip install -r /app/environment/package/syncnet/requirements.torch.txt"

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh \
    && conda activate /app/environment/syncnet \
    && pip install -r /app/environment/package/syncnet/requirements.txt \
    && pip install face-alignment"

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh \
    && conda activate /app/environment/api \
    && pip install -r /app/environment/package/api/requirements.torch.txt"

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh \
    && conda activate /app/environment/api \
    && pip install -r /app/environment/package/api/requirements.txt"

# 2. torch / cuda pip 依赖
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh \
    && conda activate /app/environment/nerffacespeech \
    && pip install -r /app/environment/package/nerffacespeech/requirements.torch.txt"

# 3. pytorch3d（高风险，单独层）
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh \
    && conda activate /app/environment/nerffacespeech \
    && conda install -y \
        https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu121_pyt212.tar.bz2"

# 4. 其余 Python 依赖
RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh \
    && conda activate /app/environment/nerffacespeech \
    && pip install -r /app/environment/package/nerffacespeech/requirements.txt"

# 5. pytorch3d / nvdiffrast 编译依赖

# =====================================================================
# ==================== llm_talk 环境 ==================================
# =====================================================================

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh \
    && conda activate /app/environment/llm_talk \
    && pip install -r /app/environment/package/llm_talk/requirements.torch.txt"

RUN /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh \
    && conda activate /app/environment/llm_talk \
    && pip install -r /app/environment/package/llm_talk/requirements.txt \
    && pip install pkuseg==0.0.25 --no-build-isolation \
    && pip install chatterbox-tts==0.1.3"


RUN /bin/bash -c "set -e \
    && source /opt/conda/etc/profile.d/conda.sh \
    && conda activate /app/environment/nerffacespeech \
    && pip install -r /app/environment/package/nerffacespeech/requirements.pytorch3d.txt --no-build-isolation"
# =====================================================================
# ==================== 复制项目代码（最后） ============================
# =====================================================================

COPY . /app/

# ==================== 默认命令 ====================
CMD ["/bin/bash"]
