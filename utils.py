
import face_recognition as fr
import numpy as np
import os
import cv2

def get_sequential_optical_flow_fd(video_path ,frame_size=(256, 256), max_frames=50):
    print("Get sequential optical flow on {}".format(video_path))
    x_expand, y_expand = 1.5, 1.8

    sequential_optical_flow = []

    # extract frames from video
    frames = extract_frames(video_path)
    
    for i in range(max_frames):
        face_locations = fr.face_locations(frames[i])

        # extract the first person in the frame
        if len(face_locations) >= 1:
            top, right, bottom, left = face_locations[0]
            b, h, w, c = frames.shape

            # cropping area
            top = max(0, int((bottom+top)//2 - (bottom-top)//2*y_expand))  # max(0, int(top-1.5*expand))
            right = min(w-1, int((right+left)//2 + (right-left)//2*x_expand))  # min(w-1, right+expand)
            bottom = min(h-1, int((bottom+top)//2 + (bottom-top)//2*y_expand))  # min(h-1, int(bottom+1.5*expand))
            left = max(0, int((right+left)//2 - (right-left)//2*x_expand))  # max(0, left-expand)


            frame1 = cv2.cvtColor(cv2.resize(frames[i, top:bottom, left:right], frame_size), cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(cv2.resize(frames[i+1, top:bottom, left:right], frame_size), cv2.COLOR_BGR2RGB)

            prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
            nxt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            optical_flow = get_farneback_flow(prvs, nxt)
            sequential_optical_flow.append(optical_flow)

    return np.array(sequential_optical_flow)

def get_sequential_optical_flow_no_fd(video_path, frame_size=(256, 256), max_frames=50):
    """
    Belirtilen videodan ardışık kare çiftleri üzerinden optik akış hesaplar.
    
    Args:
        video_path (str): Video dosyasının yolu.
        frame_size (tuple): Karelerin yeniden boyutlandırılacağı hedef boyut (genişlik, yükseklik).
        max_frames (int): Kaç kare çifti üzerinden işlem yapılacağı.

    Returns:
        np.ndarray: Her bir kare çifti için (magnitude, angle) içeren optik akış dizisi.
    """
    # print(f"Calculating sequential optical flow from: {video_path}")
    
    sequential_optical_flow = []
    frames = extract_frames(video_path)

    for i in range(min(len(frames) - 1, max_frames)):
        # Kareleri yeniden boyutlandır ve gri tonlamaya dönüştür
        frame1 = cv2.resize(frames[i], frame_size)
        frame2 = cv2.resize(frames[i + 1], frame_size)

        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Optik akışı hesapla
        flow = cv2.calcOpticalFlowFarneback(
            prvs, nxt, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        # Polar koordinatlara dönüştür
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_2ch = np.array([mag, ang], dtype=np.float32)

        sequential_optical_flow.append(flow_2ch)

    return np.array(sequential_optical_flow)

def extract_frames(video_path):
    """Given the video path, extract every frame from video."""

    reader = cv2.VideoCapture(video_path)
    frameCount = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    frame_num = 0
    while reader.isOpened():
        success, image = reader.read()
        if not success:
            break
        buf[frame_num] = image
        frame_num += 1
    reader.release()

    return buf


def get_farneback_flow(prvs, nxt):
    h, w = prvs.shape
    flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    flow = np.array([mag, ang])
    flow_16bit = np.float16(flow)

    return flow_16bit