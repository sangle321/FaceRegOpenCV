import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    # Get all file in foldel dataSet
    dir_Paths = [os.path.join(path, f) for f in os.listdir(path)]
    imagePaths = []
    for i in range(len(dir_Paths)):
        for file_img in os.listdir(dir_Paths[i]):
            imagePaths.append(dir_Paths[i]+'/' + file_img)
    # create empth face list
    faceSamples = []
    # create empty ID list
    Ids = []
    # Looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        if (imagePath[-3:] == "jpg"):
            # loading the image and converting it to gray scale
            pilImage = Image.open(imagePath).convert('L')
            # Now we are converting the PIL image into numpy array
            imageNp = np.array(pilImage, 'uint8')
            # getting the Id from the image
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            # extract the face from the training image sample
            faces = detector.detectMultiScale(imageNp)
            # If a face is there then append that in the list as well as Id of it
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y+h, x:x+w])
                Ids.append(Id)
    return faceSamples, Ids

def train():
    # Get faces and id from foldel Dataset
    faceSamples, Ids = getImagesAndLabels('dataSet')

    # Train model to feature extraction of face
    recognizer.train(faceSamples, np.array(Ids))

    # Save model
    recognizer.save('recognizer/trainner.yml')

    print("Trained!")

if __name__ == '__main__':
    train()
