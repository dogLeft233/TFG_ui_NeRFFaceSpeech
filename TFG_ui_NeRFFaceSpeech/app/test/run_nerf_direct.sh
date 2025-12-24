#!/bin/bash
# 直接运行 NeRF 视频生成脚本的示例命令
# 不通过后端，直接在命令行执行

# 设置路径（使用绝对路径）
PROJECT_ROOT="/root/autodl-tmp/TFG_TALK_NeRFaceSpeech"
NERF_PYTHON="${PROJECT_ROOT}/environment/nerffacespeech/bin/python"
SCRIPT_PATH="${PROJECT_ROOT}/NeRFFaceSpeech_Code/StyleNeRF/main_NeRFFaceSpeech_audio_driven_w_given_poses_cache.py"

# 必需参数
NETWORK="${PROJECT_ROOT}/NeRFFaceSpeech_Code/pretrained_networks/ffhq_1024.pkl"
OUTDIR="${PROJECT_ROOT}/NeRFFaceSpeech_Code/outputs/video/test_direct"
TEST_DATA="${PROJECT_ROOT}/NeRFFaceSpeech_Code/outputs/audio/cda6d8aa-f8cc-45bf-afda-6e2712ac0588.wav"
TEST_IMG="${PROJECT_ROOT}/assets/charactors/Ayanami/ayanami.png"
MOTION_GUIDE="${PROJECT_ROOT}/NeRFFaceSpeech_Code/frames"  # 如果存在的话
CACHE_DIR="${PROJECT_ROOT}/assets/charactor/ayanami"

# 创建输出目录
mkdir -p "${OUTDIR}"

# 运行命令
"${NERF_PYTHON}" "${SCRIPT_PATH}" \
    --network "${NETWORK}" \
    --outdir "${OUTDIR}" \
    --test_data "${TEST_DATA}" \
    --test_img "${TEST_IMG}" \
    --motion_guide_img_folder "${MOTION_GUIDE}" \
    --cache_dir "${CACHE_DIR}" \
    --trunc 0.7 \
    --noise-mode const

echo ""
echo "视频生成完成！"
echo "输出文件: ${OUTDIR}/output_NeRFFaceSpeech.mp4"

