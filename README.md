# Abhorrent Sign Tracker - Face And Hand Detector

Abhorrent Sign Tracker is a Python application that uses computer vision techniques to perform hand and face detection in real-time using your webcam. It leverages the OpenCV library for image processing and the MediaPipe library for hand and face tracking.

## Features

- **Hand Detection:** Bird Flip can detect and track your hand in the video stream. It can identify the number of fingers you're holding up and display messages accordingly, such as "Middle Finger Shown" when you display your middle finger.

- **Face Detection:** The application also performs face detection using a pre-trained deep learning model. It can identify faces in the video stream and draw bounding boxes around them.

- **Audio Alert:** When Bird Flip detects the gesture "Middle Finger Shown," it plays an audio alert to provide additional feedback to the user.

- **Visual Feedback:** Bird Flip provides visual feedback to the user through different messages and overlays. When specific gestures are detected, such as showing the middle finger, the application displays corresponding messages on the screen.

## How to Use

1. **Install the required libraries:** Before running Bird Flip, make sure you have all the required libraries installed. You can install the necessary packages by running the following command:
```
pip install opencv-python numpy mediapipe cvzone playsound

```

2. **Download Audio File:** Download an audio file in MP3 format for the audio alert. Replace the file path in the code with the actual path to the MP3 file (variable `alert_sound` in `flip.py`).

3. **Run the Application:** Execute the `flip.py` script using the Python interpreter. The webcam will open, and you'll be able to see the real-time hand and face detection.

4. **Gesture Interaction:** Try showing your middle finger in front of the camera. The application will display the message "Middle Finger Shown" when it detects the gesture and play the audio alert.

5. **Visual Effects:** You can experiment with different visual effects and overlays. For example, modify the code to display emojis or stickers on specific facial features, apply image filters, or use augmented reality effects.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- MediaPipe
- cvzone
- playsound

## Known Issues

- The application might have reduced performance on low-end hardware due to the computational load of real-time hand and face tracking.

## Contributions

Contributions to Bird Flip are welcome! If you have any suggestions, improvements, or new visual effects to add, feel free to create a pull request.

## License

Bird Flip is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Bird Flip utilizes the power of OpenCV and MediaPipe libraries for hand and face tracking. Special thanks to the developers and contributors of these open-source projects.

