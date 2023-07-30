import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7)

# Load the face detection model
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

caught = False
caught_timer = time.time()

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, flipType=False)  # Set flipType to False for one hand detection

    if hands and len(hands) > 0:
        hand = hands[0]
        fingers = detector.fingersUp(hand)

        # Check for middle finger being shown
        if fingers[2] == 1 and sum(fingers) == 1:
            cv2.putText(img, "Middle Finger Shown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            caught = False  # Reset the "Caught" flag

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

            # Check if the fist is near the detected face (within a certain threshold distance)
            if len(hands) > 0:
                dist = ((hands[0]["center"][0] - (startX + (endX - startX)//2))**2 + (hands[0]["center"][1] - (startY + (endY - startY)//2))**2)**0.5
                if dist < 100:
                    cv2.putText(img, "Caught", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    caught = True
                    caught_timer = time.time()
        else:
            caught = False

    print("Caught:", caught)

    if caught and time.time() - caught_timer > 2:  # 2 seconds timeout
        caught = False

    if not caught:
        cv2.imshow("Hand and Face Detection", img)
    else:
        # If "Caught" is displayed, show a blank screen (optional)
        blank_img = img.copy()
        blank_img.fill(0)
        cv2.imshow("Hand and Face Detection", blank_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
