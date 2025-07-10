#!/usr/bin/env python3
import argparse
import os
import glob
import numpy as np
import multiprocessing
from tqdm import tqdm
from env import DATASET_DIR, SINGULAR_DIR, SEQ_DIR

CATEGORIES = ['original_sequences', 'manipulated_sequences']

def process_sequence(task):
    """
    Worker function to process a single .npy sequence file.
    Loads the sequence, splits it into individual frames, and saves each frame.
    """
    npy_path, singular_dir, cat, ds = task
    try:
        arr = np.load(npy_path)  # shape: (T,2,H,W)
        base = os.path.splitext(os.path.basename(npy_path))[0]
        out_base_dir = os.path.join(singular_dir, cat, ds)
        
        # This check is done here to be robust in a parallel environment
        os.makedirs(out_base_dir, exist_ok=True)

        for t in range(arr.shape[0]):
            single_frame = arr[t]  # shape: (2,H,W)
            out_path = os.path.join(out_base_dir, f"{base}_frame{t}.npy")
            np.save(out_path, single_frame)
    except Exception as e:
        print(f"Error processing {npy_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Split sequence .npy files into individual flow frames and save under singular_dir (Parallel Version)"
    )
    parser.add_argument(
        '--flow_root', default=SEQ_DIR,
        help="Root dir of seq flows: contains original_sequences/ and manipulated_sequences/"
    )
    parser.add_argument(
        '--singular_dir', default=SINGULAR_DIR,
        help="Output dir where individual flow .npy files will be saved"
    )
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help="List of dataset names to include (e.g. youtube Face2Face). Default: all under each category"
    )
    parser.add_argument(
        '--num_workers', type=int, default=64,
        help="Number of parallel worker processes to use."
    )
    args = parser.parse_args()

    # Step 1: Collect all tasks to be processed
    tasks = []
    print("Scanning for sequence files...")
    for cat in CATEGORIES:
        category_path = os.path.join(args.flow_root, cat)
        if not os.path.isdir(category_path):
            continue
            
        for ds in os.listdir(category_path):
            if args.datasets and ds not in args.datasets:
                continue
            
            seq_dir = os.path.join(args.flow_root, cat, ds)
            if not os.path.isdir(seq_dir):
                continue

            for npy_path in glob.glob(os.path.join(seq_dir, '*.npy')):
                # Each task is a tuple containing all necessary info for the worker
                tasks.append((npy_path, args.singular_dir, cat, ds))

    if not tasks:
        print("No .npy files found to process. Exiting.")
        return

    print(f"Found {len(tasks)} sequence files to process using {args.num_workers} workers.")

    # Step 2: Process tasks in parallel using a multiprocessing pool
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # Use tqdm to create a progress bar
        # list() consumes the iterator from imap_unordered to ensure all tasks are processed
        list(tqdm(
            pool.imap_unordered(process_sequence, tasks), 
            total=len(tasks),
            desc="Generating singular flows"
        ))

    print("\nSingular flow files have been generated.")
    
if __name__ == '__main__':
    main()