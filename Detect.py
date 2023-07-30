import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import pygame

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7)

# Load the face detection model
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

alert_sound = "C:/Users/SEC/Desktop/Projects/Bird_Flip/alert_sound.mp3"  # Replace with the actual path

pygame.mixer.init()
sound_played = False

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, flipType=False)  # Set flipType to False for one hand detection

    if hands and len(hands) > 0:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        # Check for middle finger being shown
        if fingers[2] == 1 and sum(fingers) == 1:
            cv2.putText(img, "Offensive Gesture Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if not sound_played:
                pygame.mixer.music.load(alert_sound)
                pygame.mixer.music.play()
                sound_played = True
        else:
            sound_played = False

    # Perform face detection using the loaded face detection model
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            h, w = img.shape[:2]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

    cv2.imshow("Hand and Face Detection", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
