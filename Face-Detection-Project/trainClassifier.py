import numpy as np
from PIL import Image
import os
import cv2


def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    for image in path:

        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')

        # Extract the ID from the filename
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)
        
    ids = np.array(ids)

    # Train the classifier and sav
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

train_classifier("data")
