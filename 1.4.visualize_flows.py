import os
import cv2
import glob
import numpy as np
import random
from tqdm import tqdm
from env import DATASET_DIR, SEQ_DIR

GRID_ROWS, GRID_COLS = 4, 4  # 8 orijinal + 8 flow
FRAME_SIZE = (256, 256)
FRAME_COUNT = 50
FPS = 25
DURATION_SECONDS = 6
REPEAT = 1

def list_all_videos(root_dir):
    allowed_dirs = ['youtube', 'Face2Face']
    pattern = os.path.join(root_dir, '*', '*', '*')
    video_paths = [
        p for p in glob.glob(pattern)
        if p.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        and any(allowed in p for allowed in allowed_dirs)
    ]
    return sorted(video_paths)


def get_flow_path(video_path):
    rel_path = os.path.relpath(video_path, DATASET_DIR)
    return os.path.join(SEQ_DIR, os.path.splitext(rel_path)[0] + '.npy')

def extract_first_n_frames(video_path, n=50, size=(256, 256)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frames.append(frame)
    cap.release()
    return frames[:n]

def flow_to_rgb_video(flows):
    if flows.shape[1] == 2:
        flows = flows.transpose(0, 2, 3, 1)  # (N, 2, H, W) → (N, H, W, 2)
    
    h, w = flows.shape[1], flows.shape[2]
    video = []

    for i in range(flows.shape[0]):
        flow = flows[i].astype(np.float32)
        mag = flow[..., 0]
        ang = flow[..., 1]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        video.append(bgr)
    return video


def make_grid_frame(frames):
    grid_frame = []
    for row in range(GRID_ROWS):
        row_frames = frames[row * GRID_COLS:(row + 1) * GRID_COLS]
        row_combined = np.hstack(row_frames)
        grid_frame.append(row_combined)
    return np.vstack(grid_frame)

def main():
    all_videos = list_all_videos(DATASET_DIR)
    selected_videos = random.sample(all_videos, 8)
    combined_clips = []

    print("Loading and processing videos...")
    for vid_path in tqdm(selected_videos):
        flow_path = get_flow_path(vid_path)
        if not os.path.exists(flow_path):
            print(f"[WARN] Flow not found for: {vid_path}")
            continue
        

        
        original_frames = extract_first_n_frames(vid_path, n=FRAME_COUNT, size=FRAME_SIZE)
        flows = np.load(flow_path)
        flow_frames = flow_to_rgb_video(flows[:FRAME_COUNT])

        if len(original_frames) < FRAME_COUNT or len(flow_frames) < FRAME_COUNT:
            print(f"[WARN] Skipping due to insufficient frames: {vid_path}")
            continue

        combined_clips.append(original_frames)
        combined_clips.append(flow_frames)

    if len(combined_clips) != 16:
        raise RuntimeError("16 video (8 original + 8 flow) sağlanamadı. Daha fazla veri gerek.")

    print("Constructing output video...")
    output_frames = [make_grid_frame([clip[i] for clip in combined_clips]) for i in range(FRAME_COUNT)]

    out_path = "output_grid_video.mp4"
    h, w, _ = output_frames[0].shape
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
    if not out.isOpened():
        raise RuntimeError("VideoWriter açılmadı.")

    for frame in tqdm(output_frames, desc="Writing video"):
        out.write(frame)

    out.release()
    print(f"✅ Video oluşturuldu: {out_path}")

if __name__ == "__main__":
    main()
