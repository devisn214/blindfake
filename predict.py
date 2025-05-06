# predict.py

from tensorflow.keras.models import load_model
from frames import extract_frames, preprocess_frame
import numpy as np
import cv2

def predict_video(video_path):
    # Load the pre-trained model
    model = load_model('deepfake_detection_model.h5')

    # Extract frames from the video
    frames = extract_frames(video_path, frame_skip=30)

    # Prepare frames for prediction
    predictions = []
    for frame in frames:
        preprocessed_frame = preprocess_frame(frame)
        prediction = model.predict(preprocessed_frame)
        predictions.append(prediction)

    # Calculate the final prediction as the average
    avg_prediction = np.mean(predictions)
    print(f"Average prediction for the video: {avg_prediction}")
    if avg_prediction > 0.5:
        print("The video is fake.")
    else:
        print("The video is real.")

# Example usage
video_path = r"C:\Users\devin\OneDrive\Desktop\test.jpeg"
predict_video(video_path)
