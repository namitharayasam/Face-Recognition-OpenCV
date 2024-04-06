#pylint:disable=no-member
import cv2 as cv
import numpy as np
import os

people = []

for name in os.listdir("/Users/namitha/Desktop/Learning/OpenCV/Faces/train"):
    people.append(name)
people.remove(".DS_Store")
print(people)

haar_cascade = cv.CascadeClassifier("haar_face.xml")

features = np.load("features.npy", allow_pickle=True)
labels = np.load("labels.npy")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")

img = cv.imread("./Faces/val/ben_afflek/5.jpg")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)
cv.waitKey(0)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)


