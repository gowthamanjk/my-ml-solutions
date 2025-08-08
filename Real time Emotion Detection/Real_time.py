import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Define the emotions and colors for the bounding boxes
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 100, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 0),
    'Sad': (255, 0, 0),
    'Surprise': (255, 255, 0),
    'Neutral': (200, 200, 200)
}

# Load the pre-trained models
face_cascade = cv2.CascadeClassifier(r'C:\Users\gowth\Desktop\Python _ ML\Real time Emotion Detection\haarcascade_frontalface_default.xml')
emotion_model = load_model(r"C:\Users\gowth\Desktop\Python _ ML\Real time Emotion Detection\face_model.h5")

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over each detected face
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Preprocess the ROI for the emotion model
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        # Predict the emotion
        prediction = emotion_model.predict(roi_gray)[0]
        emotion_index = np.argmax(prediction)
        emotion_label = emotion_labels[emotion_index]
        confidence = prediction[emotion_index] * 100

        # Get the color for the bounding box
        color = emotion_colors[emotion_label]

        # Draw the bounding box and the emotion label with confidence
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        label_text = f'{emotion_label}: {confidence:.2f}%'
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detector', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()