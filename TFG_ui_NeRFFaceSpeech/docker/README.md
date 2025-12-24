# Docker 使用说明

## 📦 构建镜像

```bash
cd docker
docker-compose build
# 或
docker build -t nerffacespeech:latest -f docker/Dockerfile ..
```

## 🚀 启动容器

### 方式 1：使用 docker-compose（推荐）

```bash
cd docker
docker-compose up -d          # 后台运行
docker-compose up             # 前台运行
docker-compose exec nerffacespeech bash  # 进入容器
```

### 方式 2：使用 docker run

```bash
docker run -it --gpus all \
  -v $(pwd)/assets/.cache:/app/assets/.cache:rw \
  -v $(pwd)/data:/app/data:rw \
  -v $(pwd)/output:/app/output:rw \
  -v $(pwd)/outputs:/app/outputs:rw \
  -v $(pwd)/database:/app/database:rw \
  -e CUDA_VISIBLE_DEVICES=0 \
  nerffacespeech:latest
```

## 📁 卷挂载说明

### 挂载的目录

| 容器路径 | 宿主机路径 | 权限 | 说明 |
|---------|-----------|------|------|
| `/app/assets/.cache` | `../assets/.cache` | rw | 模型缓存（PyTorch、HuggingFace） |
| `/app/data` | `../data` | rw | 输入数据 |
| `/app/database` | `../database` | rw | 数据库文件 |
| `/app/output` | `../output` | rw | 输出结果 |
| `/app/outputs` | `../outputs` | rw | 输出结果（备用） |
| `/app/NeRFFaceSpeech_Code` | `../NeRFFaceSpeech_Code` | ro | 核心代码（只读） |
| `/app/eval_pipline` | `../eval_pipline` | ro | 评估脚本（只读） |

### 数据持久化

- ✅ **模型缓存**：存储在 `assets/.cache/`，容器删除后仍保留
- ✅ **输出结果**：存储在 `output/` 和 `outputs/`，容器删除后仍保留
- ✅ **数据文件**：存储在 `data/` 和 `database/`，容器删除后仍保留

## 🔧 常用命令

### 查看容器状态

```bash
docker-compose ps
docker ps | grep nerffacespeech
```

### 查看容器日志

```bash
docker-compose logs -f nerffacespeech
docker logs nerffacespeech
```

### 进入容器

```bash
docker-compose exec nerffacespeech bash
docker exec -it nerffacespeech bash
```

### 停止容器

```bash
docker-compose down
docker stop nerffacespeech
```

### 删除容器

```bash
docker-compose down -v  # 同时删除卷（谨慎使用）
docker rm nerffacespeech
```

## 🐍 使用 Conda 环境

容器内已安装 4 个 conda 环境：

1. **nerffacespeech** - 主要环境
2. **api** - API 服务环境
3. **syncnet** - SyncNet 评估环境
4. **llm_talk** - LLM 对话环境

### 激活环境

```bash
# 进入容器后
source /opt/conda/etc/profile.d/conda.sh
conda activate /app/environment/nerffacespeech
# 或
conda activate /app/environment/syncnet
```

## 📝 注意事项

1. **首次运行**：需要确保宿主机目录存在，否则会自动创建
2. **权限问题**：确保挂载目录有正确的读写权限
3. **GPU 支持**：需要安装 nvidia-docker2 或使用 `--gpus all`
4. **缓存目录**：`assets/.cache/` 可能占用大量空间（几GB到几十GB）

## 🔍 故障排查

### 容器无法启动

```bash
# 查看日志
docker-compose logs nerffacespeech

# 检查镜像是否存在
docker images | grep nerffacespeech

# 检查端口占用
docker ps -a
```

### 数据无法访问

```bash
# 检查挂载点
docker inspect nerffacespeech | grep Mounts

# 检查目录权限
ls -la ../assets/.cache
```

### GPU 不可用

```bash
# 检查 nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# 检查容器 GPU
docker exec nerffacespeech nvidia-smi
```

