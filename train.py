from frames import prepare_dataset
from model import create_model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
import numpy as np
import os
import cv2
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image

# 1. Custom data generator
class FrameDataGenerator(Sequence):
    def __init__(self, image_paths, labels, batch_size=32, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.load_and_preprocess_image(p) for p in batch_paths]
        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.image_paths, self.labels))
            np.random.shuffle(temp)
            self.image_paths, self.labels = zip(*temp)

    def load_and_preprocess_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (299, 299))
        img = image.img_to_array(img)
        img = preprocess_input(img)
        return img

# 2. Train the model using data generators
def train_model(model, train_gen, val_gen, epochs=30):
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, verbose=1)
    return history

# 3. Evaluate model using generator
def test_model(model, test_gen):
    test_loss, test_acc = model.evaluate(test_gen, verbose=2)
    print(f'\nTest Accuracy: {test_acc * 100:.2f}%')
    print(f'Test Loss: {test_loss:.4f}')

# 4. Main driver function
def main():
    real_video_folder = r"data\real"
    fake_video_folder = r"data\fake"
    
    real_videos = [os.path.join(real_video_folder, f) for f in os.listdir(real_video_folder)]
    fake_videos = [os.path.join(fake_video_folder, f) for f in os.listdir(fake_video_folder)]
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(real_videos, fake_videos, frame_skip=30)

    # Create generators instead of loading full data into memory
    train_gen = FrameDataGenerator(X_train, y_train, batch_size=32)
    val_gen = FrameDataGenerator(X_val, y_val, batch_size=32)
    test_gen = FrameDataGenerator(X_test, y_test, batch_size=32, shuffle=False)

    if os.path.exists('deepfake_detection_model.h5'):
        print("Loading pre-trained model...")
        model = load_model('deepfake_detection_model.h5')
        history = None
    else:
        print("Creating and training new model...")
        model = create_model()
        history = train_model(model, train_gen, val_gen, epochs=30)
        model.save('deepfake_detection_model.h5')

    test_model(model, test_gen)

    if history:
        print("\nTraining Accuracy:", history.history['accuracy'][-1])
        print("Validation Accuracy:", history.history['val_accuracy'][-1])
    else:
        print("\nModel was pre-trained. No training history available.")

if __name__ == "__main__":
    main()
