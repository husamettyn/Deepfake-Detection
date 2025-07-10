#!/usr/bin/env python3
"""
dataset_distribution.py

Analyzes the distribution of your sequential‐flow deepfake dataset:
  - Counts total samples in each split (train / val)
  - Reports number and percentage of Original vs. Manipulated examples
"""

import argparse
from collections import Counter
import os

# Import your dataset class
import dataloader


def main():
    parser = argparse.ArgumentParser(
        description='Show distribution of Original vs. Manipulated samples in train/val splits')
    args = parser.parse_args()

    for split in ['train', 'val']:
        dataset = dataloader.DeepfakeDataset(split)
        labels = dataset.labels  # list of 0 (original) or 1 (manipulated)
        counter = Counter(labels)
        total = len(labels)

        print(f"{split.upper()} SPLIT")
        print(f"  Total samples       : {total}")
        print(f"  Original   (label=1): {counter.get(0, 0)} ({counter.get(0, 0)/total*100:5.2f}%)")
        print(f"  Manipulated(label=0): {counter.get(1, 0)} ({counter.get(1, 0)/total*100:5.2f}%)")
        print()

if __name__ == '__main__':
    main()
