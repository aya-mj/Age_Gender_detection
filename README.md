# Age and Gender Detection

This project performs real-time age and gender detection using computer vision. It uses a pre-trained deep learning model to detect faces in a video feed (from a webcam) or from a static image file selected by the user.

## Features

- Real-time detection using the webcam (`main.py`).
- Static image processing for face detection and prediction (`main2.py`).
- Displays the predicted age and gender along with bounding boxes around detected faces.

## Requirements

- Python 3.x
- OpenCV (cv2)
- Tkinter (for file selection in `main2.py`)

## Models
The models used for age and gender detection are pre-trained deep learning models. You can download the required models from the following sources:

- Face Detection Model: OpenCV's pre-trained face detector
- Age Detection Model: AgeNet model
- Gender Detection Model: GenderNet model
Ensure you download and place these files in the project directory for them to be loaded correctly.

