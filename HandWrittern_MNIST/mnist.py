import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
# The dataset is already split into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# --- Preprocessing the Data ---

# Reshape the data for the CNN model
# CNNs expect a 4D array: (number of images, height, width, channels)
# MNIST images are 28x28, and they are grayscale (1 channel)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# Normalize the pixel values from [0, 255] to [0, 1]
# This helps the model train faster and better
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# --- Building the CNN Model ---
# A Sequential model is a simple stack of layers.
model = Sequential()

# First Convolutional layer:
# It's a key part of CNNs that detects features like edges and curves.
# 32 filters, 3x3 kernel size, 'relu' activation function
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

# Max-Pooling layer:
# It downsamples the image, reducing its size and computational cost.
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer:
# It converts the 2D data from the previous layers into a 1D vector
# so it can be fed into a regular neural network (Dense layers).
model.add(Flatten())

# Regular Neural Network (Dense layers):
# These layers classify the features extracted by the CNN.
# The 'relu' activation is common for hidden layers.
model.add(Dense(128, activation='relu'))

# Output layer:
# It has 10 neurons, one for each digit (0-9).
# The 'softmax' activation function gives a probability for each digit.
model.add(Dense(10, activation='softmax'))

# --- Compiling and Training the Model ---

# Compile the model with an optimizer, loss function, and metrics to track.
# 'adam' is an efficient optimizer, and 'sparse_categorical_crossentropy'
# is the right loss function for this type of classification problem.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model. This is the main part where the neural network learns.
# We'll train for 5 epochs (one full pass over the training data).
print("Training the model...")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
# --- Save the Model ---
model.save("digit_recognizer_model.h5")
print("Model saved to digit_recognizer_model.h5")   
# --- Evaluating the Model ---

# Evaluate the model's performance on the test data.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# --- Making and Visualizing Predictions ---

# Make predictions on a few test images
predictions = model.predict(x_test[:10])

# Visualize the predictions and actual labels
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()

for i, ax in enumerate(axes):
    predicted_label = np.argmax(predictions[i])
    actual_label = y_test[i]
    ax.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"Pred: {predicted_label}\nActual: {actual_label}",
                 color="green" if predicted_label == actual_label else "red")
    ax.axis('off')

plt.tight_layout()
plt.show()