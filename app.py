from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import os
import time
import numpy as np
from utils import predict_fakeness, generate_heatmap, save_heatmap, generate_text_report, generate_audio_report
from braille_converter import convert_to_braille  # Import your Braille function here
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='frontend/static')

# Folders
UPLOAD_FOLDER = 'static'
HEATMAP_FOLDER = 'static/heatmaps'
REPORT_FOLDER = 'reports'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Make folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Load model
model = load_model('deepfake_detection_model.h5')

@app.route('/')
def home():
    # Serve the HTML frontend
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    start_time = time.time()
    frame_texts = []
    highlight_area = ""
    first_frame_saved = False
    last_frame_saved = False
    fake_scores = []

    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video
        cap = cv2.VideoCapture(file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            percent_complete = (frame_id / total_frames) * 100
            print(f"Processing frame {frame_id}/{total_frames} ({percent_complete:.2f}%)", end='\r')

            score = predict_fakeness(frame, model)
            fake_scores.append(score)

            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_texts.append(f"Frame {frame_id} ({timestamp_sec:.2f}s): {score*100:.2f}% fake probability.")

            # Save GradCAM only for first and last frame
            if frame_id == 1 and not first_frame_saved:
                heatmap = generate_heatmap(model, frame)
                heatmap_path = os.path.join(HEATMAP_FOLDER, 'first_frame_heatmap.jpg')
                save_heatmap(heatmap, frame, heatmap_path)
                highlight_area += f"First Frame Heatmap saved at: {heatmap_path}\n"
                first_frame_saved = True

            elif frame_id == total_frames and not last_frame_saved:
                heatmap = generate_heatmap(model, frame)
                heatmap_path = os.path.join(HEATMAP_FOLDER, 'last_frame_heatmap.jpg')
                save_heatmap(heatmap, frame, heatmap_path)
                highlight_area += f"Last Frame Heatmap saved at: {heatmap_path}\n"
                last_frame_saved = True

        cap.release()
        print("\nVideo analysis complete.")

        avg_score = np.mean(fake_scores)

        # Determine if the video is fake or real
        video_status = "Fake" if avg_score >= 0.5 else "Real"

        # Build report for video
        summary_text = (
            f"Analysis Summary:\n\n"
            f"The uploaded video '{filename}' was analyzed successfully.\n"
            f"It contains a total of {total_frames} frames.\n"
            f"The overall average probability of the video being fake is {avg_score*100:.2f}%. This video is considered '{video_status}'.\n\n"
        )

        detection_text = (
            "Frame-wise Analysis:\n\n"
            "Below are the fake probabilities for individual frames:\n"
            + "\n".join(frame_texts) + "\n\n"
        )

    else:  # Image
        img = cv2.imread(file_path)
        score = predict_fakeness(img, model)

        avg_score = score  # This was indented incorrectly before

        # Determine if the image is fake or real
        image_status = "Fake" if avg_score >= 0.5 else "Real"

        # Build report for image
        summary_text = (
            f"Analysis Summary:\n\n"
            f"The uploaded image '{filename}' was analyzed successfully.\n"
            f"The fakeness probability detected for this image is {score*100:.2f}%. This image is considered '{image_status}'.\n\n"
        )

        detection_text = (
            "Frame-wise Analysis:\n\n"
            "Since the input is a single image, no frame-wise analysis is available.\n\n"
        )

    end_time = time.time()
    total_time = end_time - start_time

    final_notes = (
        f"Processing Details:\n\n"
        f"The total time taken for the analysis was {total_time:.2f} seconds.\n\n"
        f"{highlight_area.strip()}\n"
    )

    full_report = summary_text + detection_text + final_notes

    # Save reports
    text_report_path = os.path.join(REPORT_FOLDER, 'report.txt')
    audio_report_path = os.path.join(REPORT_FOLDER, 'report.mp3')
    braille_report_path = os.path.join(REPORT_FOLDER, 'report_braille.txt')

    generate_text_report(full_report, text_report_path)
    generate_audio_report(full_report, audio_report_path)
    convert_to_braille(full_report, output_file=braille_report_path)

    return jsonify({
        'message': 'Detection Completed',
        'fakeness_probability': float(avg_score),       
        'text_report_path': f'/download/{os.path.basename(text_report_path)}',
        'audio_report_path': f'/download/{os.path.basename(audio_report_path)}',
        'braille_report_path': f'/download/{os.path.basename(braille_report_path)}',
        'highlight_area': highlight_area.strip(),
        'processing_time_seconds': float(total_time),
        'status': video_status if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')) else image_status  # Add status in response
    })


@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(REPORT_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
