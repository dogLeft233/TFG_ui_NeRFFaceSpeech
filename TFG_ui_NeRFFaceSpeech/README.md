# NeRFFaceSpeech_TALK

## 项目简介

NeRFFaceSpeech 是一个基于 NeRF/NeRF-like 技术与多模态生成模型的 **语音驱动人脸说话生成与评估系统**，支持从音频（和文本）驱动人脸视频合成

- **核心功能**
  - **Web UI 前端**：一键启动 Web 页面，角色训练，视频生成，角色对话。
  - **评估流水线（Eval Pipeline）**：集成 SyncNet 等工具，对生成结果进行客观指标评估，支持批量处理。

- **使用特点**
  - 同时提供 **非 Docker** 与 **Docker** 两套完整使用方式，可根据环境选择：
    - 非 Docker：适合已有深度学习开发环境的用户，可细粒度控制依赖与环境。
    - Docker：使用官方 CUDA 基础镜像和多环境打包，开箱即用，易于部署与迁移。

## 快速开始

本项目支持 **非 Docker 环境直接运行** 与 **Docker 容器化运行** 两种方式。  
建议优先阅读你准备使用的那一块。

---

### 数据准备

#### assets 下载

下载链接：`https://pan.quark.cn/s/ed10110e0807`  
下载后需解压到项目根目录，形成assets文件夹

## 非 Docker 使用

- **非 Docker 环境**（推荐在 `llm_talk` 环境下执行）：

  ```bash


  # 激活 llm_talk 环境
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate ./environment/llm_talk

  # 恢复软链接并预下载模型
  bash prepare_project.sh
  ```

- **Docker 环境**：

  使用封装脚本，在 Docker 容器内自动执行相同流程（会自动构建镜像并挂载 `assets/` 等目录）：

  ```bash

  bash docker/prepare_project_docker.sh
  ```

  该脚本会：
  - 使用 `docker/docker-compose.yml` 中的 `nerffacespeech` 服务构建/启动临时容器
  - 在容器内激活 `/app/environment/llm_talk`
  - 在 `/app` 下执行 `bash prepare_project.sh`，完成模型软链接恢复与模型预下载

#### 评估数据

下载链接`https://drive.google.com/drive/folders/1FwQoBd1ZrBJMrJE3ZzlNhK8xAe1OYGjX`
  
评估流程将使用`data/geneface_datasets/data/raw/videos`作为输入路径

### 环境安装（非 Docker）

1. **准备基础环境**
   - 操作系统：Ubuntu 22.04（推荐，其他 Linux 请自行适配）
   - GPU 驱动：支持 CUDA 12.1 的 NVIDIA 驱动
   - 已安装 `git`、`wget`、`curl`

2. **安装 Miniconda**

   ```bash
   # 下载并安装 Miniconda（64-bit Linux）
   wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
   bash /tmp/miniconda.sh -b -p $HOME/miniconda3

   # 初始化 conda（按提示执行）
   source "$HOME/miniconda3/etc/profile.d/conda.sh"
   conda config --set auto_activate_base false
   ```

3. **配置 Conda 镜像源（可选，但强烈推荐）**

   ```bash
   conda config --remove-key channels || true
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
   conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
   conda config --set channel_priority strict

   # 接受 Anaconda TOS（避免创建环境时报错）
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
   ```

4. **创建项目环境**

   仓库根目录下已经提供最小环境定义文件，推荐直接按 Docker 内相同方式创建（在项目根目录下执行）：
   ```bash

   # 1) nerffacespeech（核心生成/重建）
   conda env create -f environment/package/nerffacespeech/environment.min.yaml \
     -p environment/nerffacespeech

   # 2) api（FastAPI / TTS / LLM 等后端服务）
   conda env create -f environment/package/api/environment.min.yaml \
     -p environment/api

   # 3) syncnet（评估用 SyncNet / face-alignment）
   conda env create -f environment/package/syncnet/environment.min.yaml \
     -p environment/syncnet

   # 4) llm_talk（对话 / 文本处理 / pkuseg / chatterbox）
   conda env create -f environment/package/llm_talk/environment.min.yaml \
     -p environment/llm_talk
   ```

5. **为各环境安装 Python 依赖**

   ```bash
   # nerffacespeech: torch / CUDA 相关
   source "$HOME/miniconda3/etc/profile.d/conda.sh"
   conda activate ./environment/nerffacespeech
   pip install -r environment/package/nerffacespeech/requirements.torch.txt

   # 安装 pytorch3d（CUDA 12.1 对应版本）
   conda install -y \
     https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu121_pyt212.tar.bz2

   # 其余依赖 & 编译依赖
   pip install -r environment/package/nerffacespeech/requirements.txt
   pip install -r environment/package/nerffacespeech/requirements.pytorch3d.txt --no-build-isolation

   # api 环境
   conda activate ./environment/api
   pip install -r environment/package/api/requirements.torch.txt
   pip install -r environment/package/api/requirements.txt

   # syncnet 环境
   conda activate ./environment/syncnet
   pip install -r environment/package/syncnet/requirements.torch.txt
   pip install -r environment/package/syncnet/requirements.txt
   pip install face-alignment

   # llm_talk 环境（包含 pkuseg / chatterbox-tts）
   conda activate ./environment/llm_talk
   pip install -r environment/package/llm_talk/requirements.torch.txt
   pip install -r environment/package/llm_talk/requirements.txt
   pip install pkuseg==0.0.25 --no-build-isolation
   pip install chatterbox-tts==0.1.3
   ```

6. **恢复模型文件并预下载在线模型**

   确保已按前文说明下载并解压 `assets/` 到项目根目录，然后执行：

   ```bash
 

   # 使用 llm_talk 环境运行准备脚本
   source "$HOME/miniconda3/etc/profile.d/conda.sh"
   conda activate ./environment/llm_talk

   bash prepare_project.sh
   ```

   该脚本会：
   - 调用 `tools/manage_assets.py restore` 恢复各处模型权重（通过软链接指向 `assets/models`）
   - 调用 `tools/download_models.py` 预下载 Whisper base 与 HuggingFace 的 `ResembleAI/chatterbox` 模型

### 启动 Web UI（非 Docker）

在完成上述环境安装与数据准备后，可直接在项目根目录启动 Web UI：

```bash
cd NeRFFaceSpeech_TALK
./start.sh
```

### 评估流程（Eval Pipeline，非 Docker）

在完成环境安装、模型恢复与数据准备后，可直接使用评估脚本：

```bash
cd NeRFFaceSpeech_TALK

# 激活合适的环境（通常是 nerffacespeech 或 api，根据文档说明）
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate ./environment/nerffacespeech

bash run_eval_pipeline.sh [你的参数...]
```

更详细的评估配置（输入输出路径、批量评估示例等）请参见仓库中的 `EVAL_PIPELINE_USAGE.md`。

---

## Docker 使用

### 环境准备

1. **安装 Docker 与 GPU 运行环境**
   - 已安装 Docker（20+）和 Docker Compose（`docker compose` 子命令）
   - 已正确安装 NVIDIA Container Toolkit，下面命令能正常显示显卡：

   ```bash
   docker run --gpus all nvidia/cuda:12.1.0-devel-ubuntu22.04 nvidia-smi
   ```

2. **准备数据与模型**
   - 按前文“数据准备”下载并解压 `assets/` 到项目根目录
   - 按需要准备评估数据集到 `data/` 目录

### 构建镜像

在项目根目录执行：

```bash
cd NeRFFaceSpeech_TALK
docker compose -f docker/docker-compose.yml build nerffacespeech
```

### 使用容器执行项目准备脚本

推荐使用封装脚本，在 Docker 容器内完成模型软链接恢复与模型预下载：

```bash
cd NeRFFaceSpeech_TALK
bash docker/prepare_project_docker.sh
```

该脚本会基于 `docker/docker-compose.yml`：
- 构建/启动一次性 `nerffacespeech` 容器
- 在容器内激活 `/app/environment/llm_talk`
- 在 `/app` 下执行 `bash prepare_project.sh`

### 启动 Web UI（Docker）

推荐使用封装脚本，它会自动复用 `docker-compose` 中的 GPU / 挂载配置，并映射前端/后端端口：

```bash
cd NeRFFaceSpeech_TALK
bash docker/start_docker.sh
```

启动成功后：
- 后端 API：`http://localhost:8000`
- Web UI：`http://localhost:7860`

如果希望手动控制容器生命周期，也可以：

```bash
# 仅启动容器（默认进入 /bin/bash）
docker compose -f docker/docker-compose.yml up -d nerffacespeech

# 进入容器后手动启动 Web UI
docker exec -it nerffacespeech /bin/bash
cd /app
bash start.sh
```

### 评估流程（Eval Pipeline，Docker）

推荐使用封装脚本，它会在 Docker 容器内调用同一个 `run_eval_pipeline.sh`，并自动挂载数据/输出目录：

```bash
cd NeRFFaceSpeech_TALK
bash docker/run_eval_pipeline_docker.sh [你的参数...]
```

所有传给 `run_eval_pipeline.sh` 的参数，可以原样传给 `docker/run_eval_pipeline_docker.sh`。  
更详细的评估配置（输入输出路径、批量评估示例等）同样请参见 `EVAL_PIPELINE_USAGE.md`。