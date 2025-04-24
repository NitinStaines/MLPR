import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Set path to the CIFAR-10 dataset
DATA_PATH = r"E:\ml_practice\cifar-10-python\cifar-10-batches-py"

# Load one batch
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        data = batch[b'data']
        labels = batch[b'labels']
        return data, labels

# Load all training data
x_train = []
y_train = []

for i in range(1, 6):
    data_batch, labels_batch = load_cifar_batch(os.path.join(DATA_PATH, f"data_batch_{i}"))
    x_train.append(data_batch)
    y_train += labels_batch

x_train = np.concatenate(x_train)
y_train = np.array(y_train)

# Load test data
x_test, y_test = load_cifar_batch(os.path.join(DATA_PATH, "test_batch"))
x_test = np.array(x_test)
y_test = np.array(y_test)

# Normalize to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# -------------------- Deep Neural Network --------------------
model = models.Sequential([
    layers.Input(shape=(3072,)),                      # 32x32x3 = 3072
    layers.Dense(1024, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -------------------- Train --------------------
history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                    validation_data=(x_test, y_test))

# -------------------- Evaluate --------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")

# -------------------- Plot Accuracy/Loss --------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss")
plt.legend()

plt.show()
