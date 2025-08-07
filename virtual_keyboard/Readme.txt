Project Overview

This project is a real-time virtual keyboard that allows you to type by using hand gestures. It uses computer vision to detect your hand and finger movements via a webcam, enabling a touchless typing experience. The application displays a virtual keyboard on the screen, and you can "press" keys by moving your index finger to a letter and making a click gesture.

Key Features

  * Real-Time Hand Tracking: Uses a lightweight model to accurately detect a single hand in the webcam feed.
  * Virtual Interface: Displays a customizable on-screen keyboard that you can interact with.
  * Intuitive Controls: Allows for natural typing by detecting a "click" gesture when your index and middle fingers are brought together.
  * System Integration: Types directly into any active text field on your computer.

Technology Stack

  * Python: The core programming language for the project.
  * OpenCV: Used for handling the webcam stream and drawing the graphical user interface (the keyboard).
  * MediaPipe: A framework from Google that powers the hand detection and landmark tracking.
  * cvzone: A simplified wrapper around OpenCV and MediaPipe, making the hand tracking implementation much easier.
  * pynput: A library that allows the Python script to control the mouse and keyboard to simulate key presses.

Getting Started

 Prerequisites

To run this project, you need to have **Python 3.6** or higher installed on your system.

Installation

1.  Clone the repository or download the project files.

2.  Navigate to the project directory in your terminal.

3.  Install the required libraries using `pip`:

    ```bash
    pip install opencv-python mediapipe cvzone pynput
    ```

Running the Application

1.  Make sure your webcam is connected and not being used by any other application.

2.  Execute the main script from your terminal:

    ```bash
    python virtual_keyboard.py
    ```

3.  A window will pop up showing your webcam feed with the virtual keyboard overlay. To exit the application, press the 'q' key.

Contributing

This project is a great starting point for exploring computer vision and gesture control. Feel free to fork the repository, add new features (like more complex gestures or a customizable keyboard layout), and submit pull requests.

 License

This project is licensed under the MIT License. See the `LICENSE` file for details.