# StyleNeRF 训练指南

本文档说明如何使用 `run_train.py` 脚本进行模型训练。

## 快速开始

### 基本训练命令

```bash
python StyleNeRF/run_train.py \
    --outdir=training_outputs/my_training \
    --data=/path/to/training/images \
    --resume=pretrained_networks/ffhq_1024.pkl \
    --kimg=50 \
    --batch=4 \
    --snap=5
```

## 参数说明

### 必需参数

- `--outdir`: 训练输出目录，所有结果将保存在此目录
- `--data`: 训练数据集路径（目录或 zip 文件）

### 可选参数

#### 模型配置

- `--resume`: 从预训练模型继续训练（推荐）
- `--resume-start`: 从指定步数开始（默认: 0）
- `--model`: 模型配置名称（默认: `style_ffhq_ae_basic`）

#### 训练设置

- `--kimg`: 总训练轮数（千图像单位，默认: 50）
- `--batch`: 总批次大小（默认: 4）
- `--batch-gpu`: 每个 GPU 的批次大小（默认: 4）
- `--resolution`: 图像分辨率（默认: 1024）

#### 快照设置

- `--snap`: 网络快照保存间隔（默认: 5）
- `--imgsnap`: 图像快照保存间隔（默认: 1）

#### 数据增强

- `--aug`: 数据增强概率（`noaug` / `ada` / 浮点数，默认: `noaug`）
- `--mirror`: 启用水平翻转增强（标志）

#### 其他

- `--gpus`: 使用的 GPU 数量（默认: 1）
- `--gamma`: R1 正则化权重（默认: 10.0）
- `--seed`: 随机种子（默认: 0）

## 训练数据集格式

训练数据集应该是一个包含图像的目录：

```
training_data/
├── image1.jpg
├── image2.jpg
├── image3.png
└── ...
```

或者是一个 zip 文件：

```
training_data.zip
├── image1.jpg
├── image2.jpg
└── ...
```

## 完整示例

### 示例 1: 从预训练模型微调

```bash
python StyleNeRF/run_train.py \
    --outdir=training_outputs/ffhq_finetune \
    --data=/path/to/my/images \
    --resume=pretrained_networks/ffhq_1024.pkl \
    --kimg=100 \
    --batch=8 \
    --batch-gpu=4 \
    --snap=10 \
    --imgsnap=5 \
    --aug=ada \
    --mirror \
    --resolution=1024
```

### 示例 2: 从头开始训练（不推荐）

```bash
python StyleNeRF/run_train.py \
    --outdir=training_outputs/from_scratch \
    --data=/path/to/large/dataset \
    --kimg=25000 \
    --batch=32 \
    --batch-gpu=8 \
    --snap=50 \
    --aug=ada \
    --mirror
```

## 训练输出

训练过程中会在 `outdir` 目录下生成：

- `network-snapshot-*.pkl`: 网络模型快照
- `fakes*.png`: 生成的样本图像
- `reals.png`: 真实训练图像网格
- `fakes_init.png`: 初始生成的图像
- `stats.jsonl`: 训练统计信息（JSON Lines 格式）
- `events.out.tfevents.*`: TensorBoard 日志文件

## 监控训练

### 使用 TensorBoard

```bash
tensorboard --logdir=training_outputs/my_training
```

然后在浏览器中打开 `http://localhost:6006`

### 查看训练日志

训练日志会实时输出到控制台，包括：
- 当前训练步数（kimg）
- 损失值
- 训练速度（sec/kimg）
- GPU 内存使用情况

## 通过后端 API 训练

也可以通过后端 API 启动训练：

```bash
curl -X POST http://localhost:8000/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "/path/to/training/data",
    "base_model": "ffhq_1024.pkl",
    "kimg": 50,
    "snap": 5,
    "imgsnap": 1,
    "aug": "noaug",
    "mirror": false,
    "config_name": "style_ffhq_ae_basic"
  }'
```

查询训练状态：

```bash
curl http://localhost:8000/train/status/{task_id}
```

## 注意事项

1. **从预训练模型开始**: 强烈建议使用 `--resume` 参数从预训练模型开始训练，而不是从头开始
2. **批次大小**: 根据 GPU 内存调整 `--batch` 和 `--batch-gpu` 参数
3. **训练时间**: 训练时间取决于数据集大小、批次大小和总训练轮数
4. **GPU 内存**: 1024x1024 分辨率需要较大的 GPU 内存（建议至少 16GB）
5. **数据质量**: 确保训练图像质量良好，分辨率一致

## 故障排除

### 内存不足

- 减小 `--batch` 和 `--batch-gpu` 参数
- 降低 `--resolution`（例如 512 或 256）

### 训练速度慢

- 增加 `--batch` 和 `--batch-gpu`（如果内存允许）
- 使用多个 GPU（`--gpus`）
- 检查数据加载速度（`num_workers` 在代码中设置）

### 训练不稳定

- 降低学习率（需要修改代码中的 `G_opt_kwargs` 和 `D_opt_kwargs`）
- 调整 `--gamma` 参数
- 检查数据集质量

## 相关文件

- `run_train.py`: 训练脚本
- `training/training_loop.py`: 训练循环实现
- `training/dataset.py`: 数据集加载器
- `training/stylenerf.py`: StyleNeRF 网络定义
- `training/loss.py`: 损失函数定义

