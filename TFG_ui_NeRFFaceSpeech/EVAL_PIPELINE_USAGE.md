## 评估流程使用说明（Eval Pipeline）

本项目提供统一的评估流程脚本 `run_eval_pipeline.sh`，用于对一批视频进行切分、对齐、推理并计算 SyncNet 等指标。  
可以在 **本地非 Docker 环境** 或 **Docker 容器中** 运行，核心入口保持一致。

---

### 一、准备工作

- **数据准备**
  - 将待评估视频放到：
    - `data/geneface_datasets/data/raw/videos`
  - 支持常见视频格式（如 `.mp4` 等），脚本会自动递归处理该目录下的视频文件。

- **模型与权重**
  - 按 `README.md` 中的数据准备说明，下载并解压 `assets/` 到项目根目录。
  - 确保存在核心模型：
    - `NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl`
  - 推荐执行一次：

    ```bash
    # 使用 llm_talk 环境
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ./environment/llm_talk

    bash prepare_project.sh
    ```

    该脚本会：
    - 用软链接的方式，将各处模型权重统一指向 `assets/models`
    - 预下载 Whisper base 与 HuggingFace `ResembleAI/chatterbox` 模型（如需）

- **环境准备**
  - 按 `README.md` 中“环境安装”部分完成 4 个环境创建，至少需要 **syncnet 环境**：
    - `environment/syncnet`：用于运行 `eval_pipline` 模块和 SyncNet / face-alignment。

---

### 二、本地非 Docker 环境运行

脚本：`run_eval_pipeline.sh`（位于项目根目录）

#### 1. 默认行为（推荐）

直接在项目根目录运行：

```bash
cd /path/to/NeRFFaceSpeech
bash run_eval_pipeline.sh
```

脚本会自动：

- 使用 `environment/syncnet` 环境：
  - 自动查找并激活本机 Miniconda/Anaconda，再 `conda activate environment/syncnet`
- 使用默认配置：
  - 输入目录：`data/geneface_datasets/data/raw/videos`
  - 输出目录：`output/eval_YYYYMMDD_HHMMSS`
  - 模型路径：`NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl`
  - 每段时长：`8` 秒
  - 每视频最多段数：`8` 段（随机选取）
  - 对齐方式：`FFHQFaceAlignment`
  - 设备：`cuda`

执行过程中，会调用：

```bash
python -m eval_pipline \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --network "$MODEL_PATH" \
  --segment-sec "$SEGMENT_SEC" \
  --max-segments "$MAX_SEGMENTS" \
  --random-segments \
  --ffhq-alignment \
  --device cuda
```

评估成功后，输出目录结构类似：

- `output/eval_YYYYMMDD_HHMMSS/`
  - `videos_split/`：按段切分后的原始视频片段
  - `videos_cropped/`：对齐/裁剪后的视频
  - `videos_infer/`：推理生成的视频
  - `metrics.json`：汇总指标（如 SyncNet 分数）

#### 2. 修改默认参数

当前 `run_eval_pipeline.sh` 中的参数集中在顶部变量：

- `INPUT_DIR`
- `OUTPUT_DIR`
- `MODEL_PATH`
- `SEGMENT_SEC`
- `MAX_SEGMENTS`
- `CONDA_ENV_NAME`

如需自定义：

```bash
# 修改脚本头部变量后再运行
vim run_eval_pipeline.sh   # 或任意编辑器
bash run_eval_pipeline.sh
```

如果你希望在命令行直接覆写参数，可以根据 `eval_pipline` 的 argparse 解析方式，参考脚本中的 `python -m eval_pipline ...` 行，手动执行自定义命令。

---

### 三、Docker 环境运行

脚本：`docker/run_eval_pipeline_docker.sh`

该脚本会：

- 使用 `docker/docker-compose.yml` 中的 `nerffacespeech` 服务
- 自动构建镜像（如尚未构建）
- 自动挂载数据 / 输出 / 模型 / 缓存目录：
  - `assets/.cache` → `/app/assets/.cache`
  - `assets/models` → `/app/assets/models`
  - `data` → `/app/data`
  - `database` → `/app/database`
  - `output` → `/app/output`
  - `outputs` → `/app/outputs`
  - `NeRFFaceSpeech_Code`、`eval_pipline`、`run_eval_pipeline.sh` 等以只读方式挂载到容器中
- 在容器内执行：

  ```bash
  bash /app/run_eval_pipeline.sh "$@"
  ```

#### 使用方式

在项目根目录执行：

```bash
cd /path/to/NeRFFaceSpeech
bash docker/run_eval_pipeline_docker.sh [你的参数...]
```

- 传给 `docker/run_eval_pipeline_docker.sh` 的参数会原样传递给容器内的 `run_eval_pipeline.sh`。
- 如果你没有修改 `run_eval_pipeline.sh` 以支持命令行参数，则可以直接运行不带参数版本（使用默认配置）：

  ```bash
  bash docker/run_eval_pipeline_docker.sh
  ```

容器内的行为与本地非 Docker 运行时一致，只是依赖环境与 CUDA 驱动通过 Docker + NVIDIA runtime 提供。

---

### 四、常见问题（FAQ）

- **Q1：找不到输入目录 / 数据集不存在？**
  - 请确认已下载并放置数据到：
    - `data/geneface_datasets/data/raw/videos`
  - Docker 模式下，请确认宿主机路径与 `docker/docker-compose.yml` 中的挂载设置一致。

- **Q2：提示找不到 `ffhq_1024.pkl`？**
  - 检查：
    - `NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl` 是否存在
  - 若使用了 `assets/` + 软链接管理，请确保：
    - 已执行 `bash prepare_project.sh`
    - 软链接指向的目标文件存在于 `assets/models/...` 下

- **Q3：SyncNet / face-alignment 相关模块导入失败？**
  - 确保 **syncnet 环境** 已正确安装依赖：

    ```bash
    conda activate ./environment/syncnet
    pip install -r environment/package/syncnet/requirements.torch.txt
    pip install -r environment/package/syncnet/requirements.txt
    pip install face-alignment
    ```

- **Q4：显卡不可见或 CUDA 错误？**
  - 非 Docker：
    - 确认 `nvidia-smi` 正常，PyTorch 能检测到 GPU（`torch.cuda.is_available()` 为 True）
  - Docker：
    - 确认已安装 NVIDIA Container Toolkit
    - `docker run --gpus all nvidia/cuda:12.1.0-devel-ubuntu22.04 nvidia-smi` 能显示显卡
    - 使用 `docker compose -f docker/docker-compose.yml up` 时，`runtime: nvidia` 和 `NVIDIA_VISIBLE_DEVICES` 已正确配置

---

如需对评估配置（切分策略、对齐方式、指标计算等）做更细粒度的修改，可直接阅读并修改 `eval_pipline` 模块及 `run_eval_pipeline.sh` 中调用部分。该文档仅覆盖最常用、推荐的使用路径。 


