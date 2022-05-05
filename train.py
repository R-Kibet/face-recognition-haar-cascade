
import os
import cv2 as cv
import numpy as np


# create list of all the people in the image
p = []
for i in os.listdir(r"/root/Documents/train"):
    p.append(i)

# print(p)

Dir = r"/root/Documents/train"

cascade = cv.CascadeClassifier('face.xml')
# image array and who it belongs to

features =[]
labels = []

# loop over every folder in base folder looping of every image
def create_train():
    for person in p:
        path = os.path.join(Dir,person)
        label =p.index(person)

        # loop over every image in the folder creating a path
        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_arr = cv.imread(img_path)
            gray = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

            faces_rectangle = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rectangle:
                faces_roi = gray[y:y+h, x:x+w] # cropping out he faces
                features.append(faces_roi)
                labels.append(label)


create_train()
# print(f"length of the features = {len(features)}")
# print(f"length of labels : {len(labels)}")
print("training done")

# convert to numpy arrays
features = np.array(features,dtype="object")
labels = np.array(labels)

# features and labels to train
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer on features and labels
face_recognizer.train(features,labels)

# saving this face recognition configuration to be used anywhere without reconfiguring
face_recognizer.save("face_trained.yml")

np.save("features.npy", features)
np.save("labels.npy", labels)