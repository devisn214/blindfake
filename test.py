from frames import prepare_dataset
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import os

# Test the model
def test_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest Accuracy: {test_acc * 100:.2f}%')
    print(f'Test Loss: {test_loss:.4f}')
    return test_acc, test_loss

# Plot and save Confusion Matrix
def plot_and_save_confusion_matrix(model, X_test, y_test, save_path):
    y_pred_probs = model.predict(X_test)
    y_pred_labels = (y_pred_probs > 0.5).astype(int).flatten()

    cm = confusion_matrix(y_test, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    disp.plot(cmap=plt.cm.Blues)

    plt.title('Confusion Matrix')
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, "confusion_matrix.jpg")
    plt.savefig(filename, format='jpg')
    print(f"Confusion matrix saved as: {filename}")
    plt.show()

# Main function
def main():
    # Paths to video folders
    real_video_folder = r"data/real"
    fake_video_folder = r"data/fake"

    # Load test video file paths
    real_videos = [os.path.join(real_video_folder, f) for f in os.listdir(real_video_folder)][:50]
    fake_videos = [os.path.join(fake_video_folder, f) for f in os.listdir(fake_video_folder)][:50]

    # Prepare only test data
    _, _, X_test, _, _, y_test = prepare_dataset(real_videos, fake_videos, frame_skip=30)

    # Load the saved model
    model = load_model('deepfake_detection_model.h5')

    # Test the model
    test_model(model, X_test, y_test)

    # Save confusion matrix to "reports" folder
    plot_and_save_confusion_matrix(model, X_test, y_test, save_path="reports")

if __name__ == "__main__":
    main()
