import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Extract and save frames from video
def extract_and_save_frames(video_path, label, output_dir, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_paths = []

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_filename = f"{video_name}_frame{frame_count}.jpg"
            full_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(full_path, frame)
            saved_paths.append((full_path, label))

        frame_count += 1

    cap.release()
    return saved_paths

# 2. Prepare dataset by extracting frames from all videos and saving to disk
def prepare_dataset(real_videos, fake_videos, frame_skip=30, output_dir="extracted_frames"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Processing real videos...")
    real_data = []
    for idx, video_path in enumerate(real_videos):
        real_data.extend(extract_and_save_frames(video_path, label=0, output_dir=output_dir, frame_skip=frame_skip))
        print(f"Processed {idx + 1}/{len(real_videos)} real videos", end='\r')

    print("\nProcessing fake videos...")
    fake_data = []
    for idx, video_path in enumerate(fake_videos):
        fake_data.extend(extract_and_save_frames(video_path, label=1, output_dir=output_dir, frame_skip=frame_skip))
        print(f"Processed {idx + 1}/{len(fake_videos)} fake videos", end='\r')

    all_data = real_data + fake_data
    np.random.shuffle(all_data)

    paths, labels = zip(*all_data)
    labels = np.array(labels)

    # Split into train, val, test
    X_train, X_temp, y_train, y_temp = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print("\nDataset preparation complete.")
    return list(X_train), list(X_val), list(X_test), y_train, y_val, y_test
