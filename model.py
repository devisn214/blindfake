# model.py

from tensorflow.keras import layers, models
from tensorflow.keras.applications import Xception

# 4. Create and compile the model
def create_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)  # Binary classification (real or fake)
    model = models.Model(inputs=base_model.input, outputs=x)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
