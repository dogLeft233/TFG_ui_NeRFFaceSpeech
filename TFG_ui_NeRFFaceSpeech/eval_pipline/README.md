# 评估流程说明

## 1. 总览

给定一组原始 mp4 视频，流程按顺序完成：
1) 固定时长切分原视频。
2) 人脸检测 / 对齐 / 裁剪，得到统一分辨率的输入序列。
3) 将处理后的视频送入生成模型得到合成视频。
4) 对真值与生成视频计算质量与同步度指标。

## 2. 流程要点

- **切分**：切分一个大视频，每段 8 秒。为了节省算力，每个大视频随机挑选8段，各指标平均后作为大视频指标。
- **裁剪 / 对齐**(非常重要)：使用FFHQFaceAlignment，如果不用，结果会很抽象。对每个切分后的视频，对第一帧图像做FFHQ对齐，此时保存一个放射矩阵A，将A用于视频后续的图像来对整个视频做对齐。对齐后可保证真值与生成视频在同一坐标系。
- **推理**：使用提供的生成器权重ffhq_1024.pkl，对裁剪后的视频逐段生成。模型输入为：视频的第一帧图像和完整的音频，图像reszie到 1024*1024来适应模型要求。
- **评估**：在对齐后的真值与生成结果上计算多种客观指标，并汇总均值与标准差。

## 3. 输入与输出

- 输入：原始 mp4 视频目录。
- 中间结果：
  - `videos_split/`：按段切分后的片段。
  - `videos_cropped/`：对齐或裁剪后的视频。
  - `videos_infer/`：模型生成的视频。
- 输出：`metrics.json`，包含逐视频与分组汇总的指标。

## 4. 指标定义（核心公式）

- **PSNR**（峰值信噪比）  
  $$
  \text{PSNR} = 10 \log_{10}\!\left(\frac{MAX^2}{\text{MSE}}\right),\quad MAX=255
  $$
  其中 MSE 为逐像素均方误差。

- **SSIM**（结构相似性）  
  $$
  \text{SSIM}(x,y)=\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2 + C_1)(\sigma_x^2+\sigma_y^2 + C_2)}
  $$

- **FID**（Fréchet Inception Distance）  
  基于 InceptionV3 pool3 特征的均值与协方差：  
  $$
  d^2 = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
  $$

- **NIQE**（无参考质量）  
  使用 NIQE 模型评估自然图像统计偏离度。

- **LSE-C / LSE-D**（Lip-Sync Error）  
  使用Wav2Lip中的标准实现，度量音视频同步的相关性与偏移。

## 5. 典型参数（概念说明）

- `segment-sec`：切分段长（秒）。
- `max-segments` 与 `random-segments`：限制并可随机选取片段。
- `face-ratio`、`output-size`：裁剪区域与输出分辨率控制。
- `gen-res`：生成模型期望分辨率。
- `batch-size`：FID 计算批大小。
- `max-frames`：评估时的最大帧数（可下采样）。

## 6. 运行顺序（逻辑）

1) 切分：生成 `videos_split/`。  
2) 裁剪 / 对齐：生成 `videos_cropped/`。  
3) 推理：读取裁剪结果，生成 `videos_infer/`。  
4) 评估：以对齐后的真值与生成结果计算并写出 `metrics.json`。

## 7. 结果解读

- JSON 包含逐视频指标、按基础名分组的均值/标准差，以及整体均值/标准差。
