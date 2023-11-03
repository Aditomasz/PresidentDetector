import cv2
import numpy as np
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

count = 0

for filename in os.listdir('./images/Mass'):
    file_path = os.path.join('./images/Mass', filename)
    image = cv2.imread(file_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))

    if len(faces) > 0:
        x, y, w, h = faces[0]
        img = image[y:y + h, x:x + w]
        img = cv2.resize(img, (200, 200),
                            interpolation=cv2.INTER_LINEAR)
        cv2.imshow('twarz', img)

        for x, y, w, h in faces:
            count = count + 1
            name = './images/Mass results/' + str(8000 + count) + '.jpg'
            print("Creating Images........." + name)
            face_img = image[y:y + h, x:x + w]
            resized_face_img = cv2.resize(face_img, (200, 200))
            cv2.imwrite(name, resized_face_img)

