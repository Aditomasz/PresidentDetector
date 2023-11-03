import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing import image
from keras.utils import to_categorical
from tensorflow.keras.preprocessing import image

images_path = 'images'
labels_path = 'labels.npy'

images = []
labels = []

class_indices = {}
for i, class_name in enumerate(os.listdir(images_path)):
    if os.path.isdir(os.path.join(images_path, class_name)):
        class_indices[class_name] = i

for sub_dir in os.listdir(images_path):
    sub_dir_path = os.path.join(images_path, sub_dir)

    if not os.path.isdir(sub_dir_path):
        continue

    label = class_indices[sub_dir]

    for filename in os.listdir(sub_dir_path):
        file_path = os.path.join(sub_dir_path, filename)

        img = image.load_img(file_path, target_size=(200, 200), color_mode='grayscale')
        img_arr = image.img_to_array(img)
        images.append(img_arr)

        labels.append(label)

images = np.array(images)

num_classes = len(os.listdir(images_path))
labels = to_categorical(labels, num_classes=num_classes)

num_classes = len(os.listdir(images_path))
print("Number of classes: ", num_classes)

print("Class indices: ", class_indices)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(200, 200, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(images, labels, epochs=10, validation_split=0.2)

model.save('Tactical_model.h5'.format(1))