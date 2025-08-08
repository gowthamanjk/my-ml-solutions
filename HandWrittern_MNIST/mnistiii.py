import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("digit_recognizer_model.h5")

# Create a blank canvas
canvas = np.zeros((400, 400), dtype="uint8")
window_name = "Draw a Digit"
cv2.namedWindow(window_name)

# Mouse callback function to draw on the canvas
is_drawing = False
def draw_on_canvas(event, x, y, flags, param):
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        # Draw a thick white line
        cv2.circle(canvas, (x, y), 20, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False

# Set the mouse callback function
cv2.setMouseCallback(window_name, draw_on_canvas)

print("Draw a number from 0-9 on the canvas.")
print("Press 's' to save the drawing and get a prediction.")
print("Press 'c' to clear the canvas.")
print("Press 'q' to quit.")

while True:
    # Display the canvas
    cv2.imshow(window_name, canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Preprocess the drawn image for the model
        drawn_image = cv2.resize(canvas, (28, 28))
        drawn_image = drawn_image.astype('float32') / 255.0
        drawn_image = drawn_image.reshape(1, 28, 28, 1)

        # Make a prediction
        prediction = model.predict(drawn_image)
        predicted_digit = np.argmax(prediction)
        
        # Display the result
        print(f"I predict the digit is: {predicted_digit}")

    elif key == ord('c'):
        # Clear the canvas
        canvas.fill(0)
        print("Canvas cleared.")

    elif key == cv2.EVENT_FLAG_LBUTTON:
        is_drawing = False
    
    elif key == ord('q'):
        break

cv2.destroyAllWindows()