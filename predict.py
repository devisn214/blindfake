from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Load model once
model = load_model('deepfake_detection_model.h5')

# Preprocess single frame
def preprocess_frame(frame):
    frame = cv2.resize(frame, (299, 299))  # Changed from 224x224 to 299x299
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame


# Extract frames from video
def extract_frames(video_path, frame_skip=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    return frames

# Prediction function
def predict_input(path):
    ext = os.path.splitext(path)[1].lower()

    if ext in ['.jpg', '.jpeg', '.png']:
        print("Detected image input.")
        frame = cv2.imread(path)
        if frame is None:
            print("Error: Could not read the image.")
            return
        processed = preprocess_frame(frame)
        prediction = model.predict(processed)[0][0]
        print(f"Prediction: {prediction:.4f}")
        print("The image is fake." if prediction > 0.5 else "The image is real.")

    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        print("Detected video input.")
        frames = extract_frames(path, frame_skip=30)
        if not frames:
            print("No frames extracted. Check the video path or content.")
            return

        predictions = []
        for frame in frames:
            processed = preprocess_frame(frame)
            prediction = model.predict(processed)[0][0]
            predictions.append(prediction)

        avg_prediction = np.mean(predictions)
        print(f"Average prediction for the video: {avg_prediction:.4f}")
        print("The video is fake." if avg_prediction > 0.5 else "The video is real.")

    else:
        print("Unsupported file type. Please provide a valid image or video file.")

# Example usage
file_path = r"C:\Users\devin\OneDrive\Desktop\ttest.jpeg"
predict_input(file_path)
