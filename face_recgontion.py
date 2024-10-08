import numpy as np
import cv2 as cv

haar_cascade= cv.CascadeClassifier('haar_face.xml')
people= ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']


#features=np.load('features.npy')
#labels= np.load('lables.npy')

face_recognizer= cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face.trained.yml')

img= cv.imread(r'D:\open cv\Faces\Elton John\1.jpg')
gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect a face

faces_rect= haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=4)

for(x, y, w, h) in faces_rect:
    faces_region = gray[y:y+h, x:x+h]
    label, confidence= face_recognizer.predict(faces_region)
    print(f'label= {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)