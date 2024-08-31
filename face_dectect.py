import cv2 as cv

img =cv.imread('images/group 1.jpg')
#cv.imshow('lady', img)

gray=cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#cv.imshow('Gray', gray)

haar_cascade= cv.CascadeClassifier('haar_face.xml') #read all 33k line of code

face_rect= haar_cascade.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=1) #no of rectangles containing a face
print(len(face_rect))

for x,y,w,h in face_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
cv.imshow('Dectected face',img )

cv.waitKey(0)




#for face dectecton in vdo




import cv2 as cv

# Load the Haar Cascade classifier
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Open video capture (either a video file or webcam)
video_capture = cv.VideoCapture('path_to_video.mp4')  # Use '0' for webcam

while True:
    ret, frame = video_capture.read()  # Capture frame-by-frame
    
    if not ret:
        break
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert frame to grayscale
    
    face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    
    for x, y, w, h in face_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    
    cv.imshow('Video - Face Detection', frame)  # Display the frame with detected faces
    
    if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release the video capture object and close windows
video_capture.release()
cv.destroyAllWindows()
