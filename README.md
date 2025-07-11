# Deepfake Detection Using Optical Flow

> This README was generated with AI assistance.

This repository implements a deep learning pipeline to detect deepfake videos using optical flow analysis. The architecture combines a **VGG11** network as a feature extractor for optical flow frames and a **Self-Attention** mechanism to model temporal dependencies across frame sequences. For more information visit original repository: https://github.com/cyshih704/Deepfake-Detection

> **IMPORTANT CHANGE:** This project originally performed face detection and cropping before extracting optical flow sequences. In this version, the face detection feature has been disabled. However, it can be easily reactivated by modifying the function call on line 23 of the `1.1.generate_sew_flows.py` file. By replacing the `get_sequential_optical_flow_no_fd` function with `get_sequential_optical_flow_fd`, the face detection functionality will be enabled again. `fd` stands for Face Detection.

---

## Changelog:

### v2:

**feature:**

* changed optical flow generation logic; added generation of optical flow without face detection.

### v1:

**fix:**

* fixed dataloader and reduced complexity
* simplified optical flow generation
* made training scripts easier to use
* renamed files
* deleted unnecessary files and folders
* enhanced `README.md` file

**feature:**

* added flow visualizing tool (creates a video)
* dataset checking feature
* added dataset organization scripts

---

## 📋 Project Workflow

The detection pipeline progresses through the following stages:

1. **Dataset Preparation**: Download and organize FaceForensics++ dataset.
2. **Preprocessing**:

   * Compute sequential optical flows for each video.
   * Split these into individual (singular) optical flow frames.
3. **Model Training** (Two-Stage Training):

   * **Stage 1**: Train a VGG11 model on singular flow frames.
   * **Stage 2**: Use the VGG11 encoder with a Self-Attention decoder trained on sequential flows.
4. **Inference**: Predict whether a given video is real or fake.

---

## ⚙️ Prerequisites and Setup

### 1. Clone the repository

```bash
git clone https://github.com/husamettyn/Deepfake-Detection
cd Deepfake-Detection
```

### 2. Create Conda Environment & Install Dependencies

```bash
conda env create -f environment.yml
conda activate deepfake-detection
```

> ⚠️ The code is tested under **CUDA 12.4** on Linux. Other CUDA versions may require dependency adjustments.

---

### 3. Configure Environment Paths

Edit the `env.py` file and set:

* `DATASET_DIR`: Directory for original FaceForensics videos
* `SEQ_DIR`: Directory to save sequential optical flows
* `SINGULAR_DIR`: Directory for singular optical flows

Use absolute paths if necessary.

---

> For Inference: you may proceed directly to Step 3

---

## 📚 Step 0: Dataset Setup

1. Download the dataset from the Kaggle link in `dataset/dataset.txt`.
2. Extract it under `dataset/Faceforensics/`
3. Run the following scripts:

```bash
# 1. Split dataset into original and manipulated
chmod +x dataset/organize.sh
./dataset/organize.sh dataset/Faceforensics

# 2. Flatten nested folder structure
chmod +x dataset/fix_str.sh
./dataset/fix_str.sh dataset/Faceforensics

# 3. Inspect the structure
chmod +x dataset/show_structure.sh
./dataset/show_structure.sh dataset/Faceforensics
```

### Folder Structure After Processing:

```
Faceforensics/
├── original_sequences/
│   └── youtube/                # 1000 real videos
└── manipulated_sequences/
    ├── DeepFakeDetection/     # 1000 fake videos
    ├── Deepfakes/
    ├── Face2Face/
    ├── FaceShifter/
    ├── FaceSwap/
    └── NeuralTextures/
```

---

## 🛠️ Step 1: Preprocessing

> Note: Script 1.1 and 1.2 has parallel processing, thus you may want to change worker count using --num_workers parameter.

### 1.1 Generate Sequential Optical Flows

```bash
python 1.1.generate_seq_flows.py --datasets youtube Face2Face
```

**Key arguments:**

* `--input_root`: Defaults to `DATASET_DIR` from `env.py`
* `--output_root`: Defaults to `SEQ_DIR`
* `--datasets`: Optionally specify sub-datasets
* `--max_frames`: Defaults to 50
* `--num_workers`: Defaults to 64

Each video will be represented by a `.npy` array of shape `(T, 2, H, W)`.

---

### 1.2 Split into Singular Optical Flows

```bash
python 1.2.generate_singular_flows.py
```

**Key arguments:**

* `--flow_root`: Defaults to `SEQ_DIR`
* `--singular_dir`: Defaults to `SINGULAR_DIR`
* `--num_workers`: Defaults to 64

This creates individual `.npy` files for each flow frame (`2, H, W`), used for VGG11 training.

### 1.3 Dataset Summary Check

Check dataset distribution after Step 1.1:

```bash
python 1.3.dataset_check.py
```

Outputs video counts and class distributions. Useful for validation.

### 1.3 Visualize Flows

Visualizes random 8 samples from samples and shows flows in human readable format

```bash
python 1.4.visualize_flows 
```

---

## 🧠 Step 2: Model Training

### 2.1 Train VGG11 Feature Extractor (Singular Flows)

```bash
python 2.1.train_vgg11.py --save_dir saved_vgg11
```

**Arguments:**

* `--singular_dir`: Defaults to `SINGULAR_DIR`
* `--save_dir`: Model output directory (default: `saved_vgg11`)
* `--batch_size`: Default is 48
* `--epochs`: Default is 50
* `--lr`: Default is 1e-4

> ✅ Labeling Convention:
> `original_sequences` (real) → `1`
> `manipulated_sequences` (fake) → `0`

The best model is saved as:

```text
saved_vgg11/best_vgg11.pth
```

---

### 2.2 Train Self-Attention Decoder (Sequential Flows)
> Don't forget to load if you trained a VGG11 model.

```bash
python 2.2.train_seq.py \
    --saved_model_name "self_attn_final" \
    --load_model "saved_vgg11/best_vgg11.pth"
```

**Arguments:**

* `--load_model`: Path to pretrained VGG11 model
* `--saved_model_name`: Output prefix (e.g., `"self_attn_final"`)
* `--finetune`: Optional flag to allow encoder fine-tuning
* `--batch_size`: Default is 16
* `--epoch`: Default is 1000

> ✅ Labeling Convention:
> `original_sequences` (real) → `1`
> `manipulated_sequences` (fake) → `0`

Saved models:

* `trained_model/self_attn_final.tar` (decoder)
* `trained_model/self_attn_final_encoder.tar` (encoder)

---

## 🔍 Step 3: Inference
* Inference-only users can skip training and download pretrained models. See `saved_model/download_link.txt`. Place trained models in `saved_model/`:
* `encoder.tar`
* `decoder.tar`

Run inference:

```bash
python 3.1.inference.py -i /path/to/your/video.mp4
```

> ✅ Interpretation:

* Output score **close to `0.0`** → **Fake**
* Output score **close to `1.0`** → **Real**

> ⚠️ Note: If `preds > 0.5` → prediction is **Real** (since `1 = Real`).
> Confirm this logic in the script.

---

## 📝 Notes

* Don't forget to make shell scripts executable (`chmod +x ...`).
* Use `env.py` for centralized path configuration.
