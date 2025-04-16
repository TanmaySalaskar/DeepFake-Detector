import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# ✅ Enable mixed precision ONLY if GPU supports it
if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# ✅ Enable XLA (JIT) Compilation for Faster Computation
tf.config.optimizer.set_jit(True)

# Define dataset path
DATASET_PATH = r"C:\Users\shweta\OneDrive\Desktop\DeepFake\dataset\images"
IMAGE_SIZE = (224, 224)  # Reduced size for faster training
BATCH_SIZE = 16  
EPOCHS = 50  

# ✅ Data Augmentation & Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# ✅ Load Pretrained MobileNetV2 (Faster than Xception)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# ✅ Unfreeze Only Last 10 Layers (Faster & More Efficient)
for layer in base_model.layers[:-10]:
    layer.trainable = False

# ✅ Add Custom Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)  # Helps prevent overfitting
output = Dense(1, activation="sigmoid")(x)  # Binary Classification (Real vs Fake)

# ✅ Create Model
model = Model(inputs=base_model.input, outputs=output)

# ✅ Use Adaptive Learning Rate Scheduler
optimizer = Adam(learning_rate=0.001)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

# ✅ Compile Model
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# ✅ Train Model (with Multi-processing for Faster Loading)
# Train the model (REMOVE workers)
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)


# ✅ Save Trained Model
model.save("models\images_model.h5")
print("✅ Faster model training complete! Saved as deepfake_detector_fast.h5")
