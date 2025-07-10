#!/usr/bin/env python3
"""
generate_seq_flows_parallel.py

Compute sequential optical flow (no face detection) for each specified subset of
videos under your Faceforensics dataset in parallel and save as .npy arrays.
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils import get_sequential_optical_flow_no_fd
from env import SEQ_DIR, DATASET_DIR

CATEGORIES = ['original_sequences', 'manipulated_sequences']
NUM_WORKERS = 64  # Adjust this based on actual system capability

def process_video(task):
    cat, dataset, vid, input_root, output_root, max_frames = task
    video_path = os.path.join(input_root, cat, dataset, vid)

    flows = get_sequential_optical_flow_no_fd(
        video_path,
        frame_size=(256, 256),
        max_frames=max_frames
    )

    if flows is None or flows.size == 0:
        return None

    if np.isinf(flows).any() or np.isnan(flows).any():
        print(f"[!] Warning: NaN/Inf in {video_path}. Cleaning...")
        flows = np.nan_to_num(flows, nan=0.0, posinf=1000.0, neginf=-1000.0)

    out_dir = os.path.join(output_root, cat, dataset)
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(vid)[0]
    out_path = os.path.join(out_dir, f"{base_name}.npy")
    np.save(out_path, flows)
    return vid

def gather_tasks(input_root, output_root, max_frames, include_datasets=None):
    tasks = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(input_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for dataset in os.listdir(cat_dir):
            if include_datasets and dataset not in include_datasets:
                continue
            vid_dir = os.path.join(cat_dir, dataset)
            if not os.path.isdir(vid_dir):
                continue
            for vid in os.listdir(vid_dir):
                if not vid.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    continue
                tasks.append((cat, dataset, vid, input_root, output_root, max_frames))
    return tasks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate .npy flow sequences from Faceforensics videos (parallel)"
    )
    parser.add_argument("--input_root", default=DATASET_DIR)
    parser.add_argument("--output_root", default=SEQ_DIR)
    parser.add_argument("--max_frames", type=int, default=50)
    parser.add_argument("--datasets", nargs='+', default=None)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="Number of parallel workers")
    args = parser.parse_args()

    tasks = gather_tasks(
        args.input_root,
        args.output_root,
        args.max_frames,
        include_datasets=args.datasets
    )
    total_videos = len(tasks)

    if total_videos == 0:
        print("No videos found to process with the given filters.")
        exit(0)

    print(f"Processing {total_videos} videos using {args.num_workers} workers...")

    successful = 0
    with Pool(processes=args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_video, tasks), total=total_videos, desc="Videos processed"):
            if result is not None:
                successful += 1

    print(f"Completed. {successful}/{total_videos} videos processed and saved successfully.")
