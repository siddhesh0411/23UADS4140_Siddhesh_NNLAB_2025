# alzheimers_vgg16_classifier.py

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

# Set dataset path
data_path = r"C:\Users\LENOVO\Desktop\Python\Datasets\Alzeimehers\Data"

# Image data generator with augmentation and validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)

# Load training data
train_generator = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=128,
    class_mode='binary',
    subset='training'
)

# Load validation data
val_generator = datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=128,
    class_mode='binary',
    subset='validation'
)

# Display class indices
print("Class indices:", train_generator.class_indices)

# Compute class weights to handle imbalance
labels = train_generator.classes
classes = list(train_generator.class_indices.values())
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(zip(classes, class_weights))
print("Class weights:", class_weight_dict)

# Load pre-trained VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# Create final model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weight_dict
)

# Plot loss curves
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train Loss', 'Validation Loss'])
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("loss_curve.png")
plt.show()

# Plot accuracy curves
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train Accuracy', 'Validation Accuracy'])
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig("accuracy_curve.png")
plt.show()
