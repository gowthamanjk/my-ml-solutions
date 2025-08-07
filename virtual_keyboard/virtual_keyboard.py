import cv2
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Controller
import time

cap = cv2.VideoCapture(0)  # Try 0 or 1 if 2 doesn't work
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)
keyboard_controller = Controller()

keyboard_keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ["SPACE"]
]

class Button():
    def __init__(self, pos, text, size=(80, 80)):
        self.pos = pos
        self.size = size
        self.text = text

button_list = []
for i, row in enumerate(keyboard_keys):
    for j, key in enumerate(row):
        if key == "SPACE":
            button_list.append(Button((300, 500), key, size=(400, 80)))
        else:
            button_list.append(Button((100 * j + 50, 100 * i + 50), key))

# For controlling press delay
last_press_time = 0
press_delay = 1  # seconds

while True:
    success, img = cap.read()
    if not success or img is None:
        continue

    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    # Draw buttons
    for button in button_list:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

    if hands:
        lmList = hands[0]['lmList']
        l, _, _ = detector.findDistance(8, 12, img, draw=False)

        if l < 30:
            for button in button_list:
                x, y = button.pos
                w, h = button.size

                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.rectangle(img, (x, y), (x + w, y + h),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                    current_time = time.time()
                    if current_time - last_press_time > press_delay:
                        if button.text == "SPACE":
                            keyboard_controller.press(" ")
                        else:
                            keyboard_controller.press(button.text.lower())
                        last_press_time = current_time

    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
