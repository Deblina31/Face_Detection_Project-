import os
import cv2 as cv
import numpy as np

p=[]
# for i in os.listdir(r'D:\open cv\Faces'):
#     p.append(i)
# print(p)
dir = r'D:\open cv\Faces'
people= ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

features=[]
lables=[]

haar_cascade= cv.CascadeClassifier('haar_face.xml')

def create_train():
    for person in people:  #looping over people
        path= os.path.join(dir,person)
        label= people.index(person)

        for img in os.listdir(path): #looping over the img in a folder
            img_path=os.path.join(path, img)

            img_array= cv.imread(img_path)
            gray=cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect= haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for(x,y,w,h) in faces_rect:
                faces_region= gray[y:y+h, x:x+w]
                features.append(faces_region)
                lables.append(label)

create_train()
print('-----------------Training Done------------')
#print(f'Length of features list = {len(features)}')
#print(f'Length of lables list = {len(lables)}')

# Converting the list into numpy arrays
features= np.array(features, dtype='object')
lables= np.array(lables)

face_recognizer= cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features list and the lables list
face_recognizer.train(features, lables)

face_recognizer.save('face.trained.yml')
np.save('features.npy', features)
np.save('lables.npy', lables)