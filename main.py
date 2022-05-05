import  cv2  as cv
import numpy as np
import os

from train import features,labels

cascade = cv.CascadeClassifier('face.xml')

p = []
for i in os.listdir(r"/root/Documents/train"):
    p.append(i)

# features = np.array(features,allow_pickle = True)
# labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img = cv.imread(r"/root/Downloads/opencv-course/Resources/Faces/val/madonna/4.jpg")
gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
cv.imshow("person", gray)

# Detect the face in the image
faces_rectangle = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in faces_rectangle:
    faces_roi = gray[y:y + h, x:x + w]

    # prediction contains 2 fields
    label, confidence = face_recognizer.predict(faces_roi)
    print(f"Label = {p[label]} with a confidence of {confidence}")

    cv.putText(img, str(p[label]), (20,20), cv.FONT_HERSHEY_COMPLEX,1.0, (0,255,0),2,)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

cv.imshow("Detected face", img)











cv.waitKey(0)