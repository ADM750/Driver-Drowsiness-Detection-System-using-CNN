import cv2
import os
import numpy as np
import time
from tensorflow.keras.models import load_model
from pygame import mixer

# Initialize sound
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascades
face = cv2.CascadeClassifier(r'haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(r'haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(r'haar cascade files/haarcascade_righteye_2splits.xml')

# Load model
model = load_model('models/cnnCat21.h5')

# Start webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    rpred, lpred = [99], [99]

    for (x, y, w, h) in right_eye:
        r_eye = gray[y:y + h, x:x + w]
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255.0
        r_eye = r_eye.reshape(1, 24, 24, 1)  # Reshape for model
        rpred = np.argmax(model.predict(r_eye), axis=-1)
        break  # Only process the first detected eye

    for (x, y, w, h) in left_eye:
        l_eye = gray[y:y + h, x:x + w]
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255.0
        l_eye = l_eye.reshape(1, 24, 24, 1)
        lpred = np.argmax(model.predict(l_eye), axis=-1)
        break

    # Detect closed eyes
    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    cv2.putText(frame, 'Score: ' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 20:
        if not mixer.get_busy():
            sound.play()

        thicc = min(thicc + 2, 16)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
