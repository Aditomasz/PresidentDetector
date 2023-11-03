import cv2
import numpy as np
import os
from keras.models import load_model

fun = None
img = None

def main():
    global swich_lib, fun, img

    video = cv2.VideoCapture()

    model = load_model('Tactical_model.h5'.format(1))

    file = str(input("Enter the test file: "))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    video.open(file)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    images_path = 'images'

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    size = (frame_width, frame_height)

    result = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    class_indices = {}
    for i, class_name in enumerate(os.listdir(images_path)):
        if os.path.isdir(os.path.join(images_path, class_name)):
            class_indices[class_name] = i

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))

            for x, y, w, h in faces:

                face_img = frame[y:y + h, x:x + w]
                resized_face_img = cv2.resize(face_img, (200, 200))
                grayed = cv2.cvtColor(resized_face_img, cv2.COLOR_BGR2GRAY)
                grayed = np.expand_dims(grayed, axis=-1)
                grayed = np.expand_dims(grayed, axis=0)
                pred = model.predict(grayed)[0]
                class_label_index = np.argmax(pred)
                class_label_text = list(class_indices.keys())[list(class_indices.values()).index(class_label_index)]
                print('The predicted class is:', class_label_text)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, class_label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            result.write(frame)
            cv2.waitKey(10)

            if cv2.waitKey(1) == ord('q'):
                break

        else:
            break

    result.release()
    video.release()
    cv2.destroyAllWindows()
    print('Finished')


if __name__ == '__main__':
    main()