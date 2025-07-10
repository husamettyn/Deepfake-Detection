import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from env import SEQ_DIR
from tqdm import tqdm

ALL_DATASETS = ['youtube', 'Face2Face']
TRAIN_RATIO = 0.8

def get_loader(split: str, batch_size: int, shuffle: bool, num_data: int = -1):
    dataset = DeepfakeDataset(split, num_data=num_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class DeepfakeDataset(Dataset):
    """Loads sequential optical-flow .npy files for deepfake detection with stratified train/val split."""
    def __init__(self, split: str, num_data: int = -1):
        # Gather all file paths and labels
        all_paths = []
        all_labels = []

        for category, label in [
            ('original_sequences', 1),    # real → 1
            ('manipulated_sequences', 0)  # fake → 0
        ]:
            for ds in ALL_DATASETS:
                npy_dir = os.path.join(SEQ_DIR, category, ds)
                if not os.path.isdir(npy_dir):
                    continue
                files = sorted(glob.glob(os.path.join(npy_dir, '*.npy')))
                all_paths += files
                all_labels += [label] * len(files)

        if len(all_paths) == 0:
            raise RuntimeError(
                f"No .npy files found under {SEQ_DIR}/{{original_sequences,manipulated_sequences}}/*"
            )

        # Optional subsampling (random without replacement)
        if num_data != -1 and len(all_paths) > num_data:
            np.random.seed(0)
            idxs = np.random.choice(len(all_paths), size=num_data, replace=False)
            all_paths = [all_paths[i] for i in idxs]
            all_labels = [all_labels[i] for i in idxs]

        # Stratified split per class
        paths_by_label = {0: [], 1: []}
        for path, lbl in zip(all_paths, all_labels):
            paths_by_label[lbl].append(path)

        train_paths, train_labels = [], []
        val_paths, val_labels = [], []
        rng = np.random.RandomState(0)

        for lbl, paths in paths_by_label.items():
            paths = list(paths)
            rng.shuffle(paths)
            n_train = int(len(paths) * TRAIN_RATIO)
            train_subset = paths[:n_train]
            val_subset = paths[n_train:]
            train_paths += train_subset
            val_paths   += val_subset
            train_labels += [lbl] * len(train_subset)
            val_labels   += [lbl] * len(val_subset)

        # Select split
        if split == 'train':
            self.x_paths = train_paths
            self.labels  = train_labels
        elif split == 'val':
            self.x_paths = val_paths
            self.labels  = val_labels
        else:
            raise ValueError("split must be 'train' or 'val'")

        # Shuffle within the split
        combined = list(zip(self.x_paths, self.labels))
        rng2 = np.random.RandomState(1)
        rng2.shuffle(combined)
        self.x_paths, self.labels = zip(*combined)
        self.x_paths = list(self.x_paths)
        self.labels  = list(self.labels)

        print(f"The length of {split} dataset: {len(self.x_paths)}")

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx: int):
        seq = np.load(self.x_paths[idx])  # shape: (≤50, 2, 256, 256)
        lbl = self.labels[idx]
        return torch.FloatTensor(seq), torch.Tensor([lbl]).float()

class SingularFlowDataset(Dataset):
    def __init__(self, root_dir, split='train', train_ratio=0.8, transform=None):
        self.files = []
        self.labels = []
        self.transform = transform

        print(f"Loading '{split}' dataset files...")

        # Kategorilere göre dosyaları ayrı ayrı topla
        label_files = {0: [], 1: []}
        categories = [('original_sequences', 1), ('manipulated_sequences', 0)]
        for cat, label in categories:
            search_dir = os.path.join(root_dir, cat)
            sub_dirs = os.listdir(search_dir)
            for ds in tqdm(sub_dirs, desc=f'Scanning {cat}', unit='dir'):
                npy_dir = os.path.join(search_dir, ds)
                for path in glob.glob(os.path.join(npy_dir, '*.npy')):
                    label_files[label].append(path)

        # Her etiketten eşit sayıda örnek alınabilecek şekilde kırp
        min_len = min(len(label_files[0]), len(label_files[1]))
        label_files[0] = label_files[0][:min_len]
        label_files[1] = label_files[1][:min_len]

        # Shuffle & split her sınıf için ayrı ayrı yapılır
        train_files, train_labels = [], []
        val_files, val_labels = [], []

        for label in [0, 1]:
            files = label_files[label]
            np.random.seed(42)
            np.random.shuffle(files)
            split_idx = int(train_ratio * len(files))
            if split == 'train':
                selected = files[:split_idx]
            else:
                selected = files[split_idx:]

            if split == 'train':
                train_files.extend(selected)
                train_labels.extend([label] * len(selected))
            else:
                val_files.extend(selected)
                val_labels.extend([label] * len(selected))

        self.files = train_files if split == 'train' else val_files
        self.labels = train_labels if split == 'train' else val_labels

        # Sınıf dağılımını yazdır
        pos = sum(self.labels)
        neg = len(self.labels) - pos
        print(f"[{split.upper()}] Total: {len(self.labels)} | Label 1: {pos} | Label 0: {neg}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])             # (2,H,W)
        x = torch.from_numpy(arr).float()
        if self.transform:
            x = self.transform(x)
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y