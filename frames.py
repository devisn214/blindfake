
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from sklearn.model_selection import train_test_split
import os

# 1. Extract frames from a video (skip frames to save memory)
def extract_frames(video_path, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only take every 'frame_skip'-th frame
        if frame_count % frame_skip == 0:
            frames.append(frame)
        
        frame_count += 1

    cap.release()
    return frames

# 2. Preprocess the frame to match Xception input size (299x299)
def preprocess_frame(frame):
    img = cv2.resize(frame, (299, 299))  # Resize to Xception input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # Preprocessing for Xception
    return img

# 3. Prepare the dataset (50 real and 50 fake videos)
def prepare_dataset(real_videos, fake_videos, frame_skip=30):
    images = []
    labels = []

    print("Processing real videos...")
    for idx, video_path in enumerate(real_videos):
        frames = extract_frames(video_path, frame_skip=frame_skip)
        for frame in frames:
            preprocessed_frame = preprocess_frame(frame)
            images.append(preprocessed_frame)
            labels.append(0)  # Real video label

        print(f"Processed {idx + 1}/{len(real_videos)} real videos", end='\r')

    print("\nProcessing fake videos...")
    for idx, video_path in enumerate(fake_videos):
        frames = extract_frames(video_path, frame_skip=frame_skip)
        for frame in frames:
            preprocessed_frame = preprocess_frame(frame)
            images.append(preprocessed_frame)
            labels.append(1)  # Fake video label

        print(f"Processed {idx + 1}/{len(fake_videos)} fake videos", end='\r')

    # Convert to numpy arrays
    images = np.vstack(images)  # Stack into one big array
    labels = np.array(labels)

    # Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print("\nDataset preparation complete.")
    return X_train, X_val, X_test, y_train, y_val, y_test
