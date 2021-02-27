import numpy as np
import cv2 as cv

solvay = cv.imread('Images/solvay-conference.jpg')

dinesh = cv.imread('Images/2.jpg')
dinesh = cv.resize(dinesh, None, fx=0.2, fy=0.2)


group = cv.imread('Images/3.jpg')
group = cv.resize(group, None, fx=0.2, fy=0.2)

#Face detection
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

#Eye detection
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

#Smile detection
smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')



def face_detect(image):

    #face_img = img.copy()
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv.rectangle(img=image, pt1=(x,y), pt2=(x+w, y+h), color=(255,0,0), thickness=2)
        roi = image[y:y+h, x:x+w]

        #Detecting eyes in the face detected
        eyes = eye_cascade.detectMultiScale(roi, minNeighbors=10)

        #Detecting smile in the face detected
        smile = smile_cascade.detectMultiScale(roi, minNeighbors=25)

        #Drawing rectangle around Eyes
        for(ex,ey,ew,eh) in eyes:
            cv.rectangle(roi, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

        #Drawing rectangle around Smile
        for(sx,sy,sw,sh) in smile:
            cv.rectangle(roi, (sx,sy), (sx+sw, sy+sh), (0,0,255), 2)

    return image


cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    frame = face_detect(frame)
    cv.imshow('Video', frame)

    if cv.waitKey(1) == 27:
        break


#img = face_detect(group)
#cv.imshow('Image', img)
#cv.waitKey()

cap.release()
cv.destroyAllWindows()