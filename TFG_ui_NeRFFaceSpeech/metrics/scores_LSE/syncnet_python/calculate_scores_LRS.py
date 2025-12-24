#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess
import glob
import os
from tqdm import tqdm

from SyncNetInstance_calc_scores import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_root', type=str, required=True, help='');
parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='');
parser.add_argument('--reference', type=str, default="demo", help='');

opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
#print("Model %s loaded."%opt.initial_model);
path = os.path.join(opt.data_root, "*.mp4")

all_videos = glob.glob(path)

prog_bar = tqdm(range(len(all_videos)))
avg_confidence = 0.
avg_min_distance = 0.

# Store results for each video
results = []

print("\n" + "="*80)
print("计算LSE-D和LSE-C分数")
print("="*80 + "\n")

for videofile_idx in prog_bar:
	videofile = all_videos[videofile_idx]
	offset, confidence, min_distance = s.evaluate(opt, videofile=videofile)
	
	# LSE-C = confidence, LSE-D = min_distance
	lse_c = confidence
	lse_d = min_distance
	
	avg_confidence += lse_c
	avg_min_distance += lse_d
	
	# Store results
	video_name = os.path.basename(videofile)
	results.append({
		'filename': video_name,
		'lse_d': lse_d,
		'lse_c': lse_c
	})
	
	# Print individual results
	print("文件: {:50s} | LSE-D: {:8.4f} | LSE-C: {:8.4f}".format(
		video_name, lse_d, lse_c))
	
	prog_bar.set_description('Avg LSE-C: {:.4f}, Avg LSE-D: {:.4f}'.format(
		avg_confidence / (videofile_idx + 1), 
		avg_min_distance / (videofile_idx + 1)))
	prog_bar.refresh()

print("\n" + "="*80)
print("汇总结果")
print("="*80)
print("总文件数: {}".format(len(all_videos)))
print("平均 LSE-C (Confidence): {:.4f}".format(avg_confidence/len(all_videos)))
print("平均 LSE-D (Distance):   {:.4f}".format(avg_min_distance/len(all_videos)))
print("="*80)



