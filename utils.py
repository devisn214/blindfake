import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
import matplotlib.pyplot as plt

# Predict fakeness directly using model output
def predict_fakeness(frame, model):
    img = cv2.resize(frame, (299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array, verbose=0)
    fake_score = prediction[0][0]  # Assuming model output shape is (1,1) or (1,2) for real/fake
    return fake_score  # Directly return probability

# Grad-CAM generation
def generate_heatmap(model, frame):
    img = cv2.resize(frame, (299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    last_conv_layer = model.get_layer('block14_sepconv2_act')
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        class_output = predictions[:, class_idx]

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

def save_heatmap(heatmap, frame, heatmap_path):
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
    cv2.imwrite(heatmap_path, superimposed_img)

# Simple text and audio report
def generate_text_report(text, path):
    with open(path, 'w') as f:
        f.write(text)

def generate_audio_report(text, path):
    import pyttsx3
    engine = pyttsx3.init()
    engine.save_to_file(text, path)
    engine.runAndWait()
