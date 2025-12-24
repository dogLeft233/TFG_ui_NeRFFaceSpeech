#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess, pickle, os, gzip, glob, sys

from SyncNetInstance_calc_scores import *

# ==================== PARSE ARGUMENT ====================

parser = argparse.ArgumentParser(description = "SyncNet");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--data_dir', type=str, default='data/work', help='');
parser.add_argument('--videofile', type=str, default='', help='');
parser.add_argument('--reference', type=str, default='', help='');
parser.add_argument('--output_file', type=str, default='', help='Output file to save scores (optional, if not specified, print to stdout)');
opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))


# ==================== LOAD MODEL AND FILE LIST ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
#print("Model %s loaded."%opt.initial_model);

flist = glob.glob(os.path.join(opt.crop_dir,opt.reference,'0*.avi'))
flist.sort()

# Check if any cropped videos were found
if not flist:
    crop_dir_path = os.path.join(opt.crop_dir, opt.reference)
    error_msg = (
        f"ERROR: No cropped video files found in {crop_dir_path}\n"
        f"Expected pattern: {os.path.join(opt.crop_dir, opt.reference, '0*.avi')}\n"
        f"\n"
        f"Possible reasons:\n"
        f"  1. Preprocessing (run_pipeline.py) has not been run yet\n"
        f"  2. Preprocessing failed or did not detect any faces\n"
        f"  3. No face tracks met the minimum length requirement (min_track)\n"
        f"\n"
        f"Please run preprocessing first:\n"
        f"  python run_pipeline.py --videofile <video.mp4> --reference {opt.reference} --data_dir {opt.data_dir}"
    )
    print(error_msg)
    sys.exit(1)

print(f"Found {len(flist)} cropped video file(s) to process")

# ==================== GET OFFSETS ====================

dists = []
results = []
for idx, fname in enumerate(flist):
    offset, conf, dist = s.evaluate(opt,videofile=fname)
    result_line = str(dist)+" "+str(conf)
    print(result_line)
    results.append(result_line)
    dists.append(dist)
      
# ==================== PRINT RESULTS TO FILE ====================

# Save to output file if specified
if opt.output_file:
    output_path = opt.output_file
    # If output_file is a relative path, save to scores_LSE directory
    if not os.path.isabs(output_path):
        # Get the directory of this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_path)
    
    # Create directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create if directory path exists
        os.makedirs(output_dir, exist_ok=True)
    
    # Append results to file
    with open(output_path, 'a') as f:
        for line in results:
            f.write(line + '\n')
    print(f"Scores saved to: {output_path}")

# Also save to work_dir if reference is provided
if opt.reference:
    work_score_file = os.path.join(opt.work_dir, opt.reference, 'scores.txt')
    os.makedirs(os.path.dirname(work_score_file), exist_ok=True)
    with open(work_score_file, 'w') as f:
        for line in results:
            f.write(line + '\n')
    print(f"Scores also saved to: {work_score_file}")

#with open(os.path.join(opt.work_dir,opt.reference,'activesd.pckl'), 'wb') as fil:
#    pickle.dump(dists, fil)
