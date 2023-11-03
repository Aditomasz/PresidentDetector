import cv2
import numpy as np
import os


fun = None
img = None


def datacollect():
    global swich_lib, fun, img
    video = cv2.VideoCapture()

    count = 0

    nameID = str(input("Enter the name: "))

    video.open(nameID + '.mp4')

    path = 'images/' + nameID

    isExist = os.path.exists(path)

    if isExist:
        pass
    else:
        os.makedirs(path)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    while True:
        ret, frame = video.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                img = frame[y:y+h, x:x+w]
                img = cv2.resize(img, (200, 200),
                                 interpolation=cv2.INTER_LINEAR)
                cv2.imshow('twarz', img)

            for x, y, w, h in faces:
                count = count + 1
                name = './images/' + nameID + '/' + str(5000 + count) + '.jpg'
                print("Creating Images........." + name)
                face_img = frame[y:y + h, x:x + w]
                resized_face_img = cv2.resize(face_img, (200, 200))
                cv2.imwrite(name, resized_face_img)

        key = cv2.waitKey(1)

        if key == 27:
            cv2.destroyAllWindows()
            break

        if count >= 1000:
            break


if __name__ == '__main__':
    datacollect()