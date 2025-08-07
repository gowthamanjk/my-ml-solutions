import pytesseract

# Replace this with the actual path to your tesseract.exe file
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\gowth\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

import cv2
import numpy as np
import pytesseract

# Set the path to the Tesseract executable
# Update this path with your Tesseract installation directory
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\gowth\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# Load the image
img = cv2.imread(r"C:\Users\gowth\Desktop\Python _ ML\LPR\Skoda_01.jpg")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a bilateral filter to reduce noise
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Find edges using the Canny edge detector
edged = cv2.Canny(gray, 30, 200)

# Find contours in the edged image
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

# Loop through contours to find the license plate
location = None
for contour in contours:
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        location = approx
        break

# Create a mask and crop the license plate
mask = np.zeros(gray.shape, np.uint8)
cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

# Crop the region of interest
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

# Use Tesseract to perform OCR on the cropped image
text = pytesseract.image_to_string(cropped_image, config='--psm 11')

print("Detected license plate text:", text.strip())

# Display the result
cv2.imshow('Car Image', img)
cv2.imshow('License Plate', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()