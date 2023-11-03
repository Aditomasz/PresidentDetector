import cv2
import numpy as np
import os
from keras.models import load_model

file = str(input("Enter image adress: ") + '.jpg')

model = load_model('Tactical_model.h5'.format(1))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

images_path = 'images'
class_indices = {}
for i, class_name in enumerate(os.listdir(images_path)):
    if os.path.isdir(os.path.join(images_path, class_name)):
        class_indices[class_name] = i

image = cv2.imread(file)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))

for x, y, w, h in faces:

    face_img = image[y:y + h, x:x + w]
    resized_face_img = cv2.resize(face_img, (200, 200))
    grayed = cv2.cvtColor(resized_face_img, cv2.COLOR_BGR2GRAY)
    grayed = np.expand_dims(grayed, axis=-1)
    grayed = np.expand_dims(grayed, axis=0)
    pred = model.predict(grayed)[0]
    class_label_index = np.argmax(pred)
    class_label_text = list(class_indices.keys())[list(class_indices.values()).index(class_label_index)]
    print('The predicted class is:', class_label_text)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(image, class_label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('Result', image)
cv2.imwrite('Result.jpg', image)

while True:
    if cv2.waitKey(1) == ord('q'):
        break
