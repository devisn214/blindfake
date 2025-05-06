from frames import prepare_dataset
from model import create_model
from tensorflow.keras.models import load_model
import os

# 5. Train the model with the dataset
def train_model(model, X_train, y_train, X_val, y_val, batch_size=8, epochs=30):
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
                        validation_data=(X_val, y_val), verbose=1)
    return history

# 6. Test the model (evaluate on a test set)
def test_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest Accuracy: {test_acc * 100:.2f}%')
    print(f'Test Loss: {test_loss:.4f}')

def main():
    # Folder paths where real and fake videos are stored
    real_video_folder = r"data\real"
    fake_video_folder = r"data\fake"
    
    # Get the file paths of real and fake videos
    real_videos = [os.path.join(real_video_folder, f) for f in os.listdir(real_video_folder)][:50]
    fake_videos = [os.path.join(fake_video_folder, f) for f in os.listdir(fake_video_folder)][:50]
    
    # Prepare the dataset (this function will not run again once the dataset is saved)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(real_videos, fake_videos, frame_skip=30)

    # Check if model exists
    if os.path.exists('deepfake_detection_model.h5'):
        print("Loading pre-trained model...")
        model = load_model('deepfake_detection_model.h5')
        # Since we are loading a pre-trained model, history is not available
        history = None  # Set history to None because no training occurred
    else:
        print("Creating and training new model...") 
        model = create_model()
        history = train_model(model, X_train, y_train, X_val, y_val, batch_size=8, epochs=30)

        # Save the model after training
        model.save('deepfake_detection_model.h5')

    # Test the model (evaluate on the test set)
    test_model(model, X_test, y_test)

    # Print final accuracies if training occurred
    if history:
        print("\nTraining Accuracy:", history.history['accuracy'][-1])
        print("Validation Accuracy:", history.history['val_accuracy'][-1])
    else:
        print("\nModel was pre-trained. No training history available.")

if __name__ == "__main__":
    main()
