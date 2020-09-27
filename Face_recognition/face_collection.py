#Import necessary Libraries
import cv2
import numpy as np

#Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Function to extract face
def face_extractor(img):
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    #Crop all faces found
    for (x,y,w,h) in faces:
        #x= x-10
        y= y-10
        cropped_face = img[y:y+h+20, x:x+w+20]

    return cropped_face

#Initialize Webcam
cap = cv2.VideoCapture(0)
count =0

#Collecting samples of faces
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame), (400,400))

        #Save file in specified directory
        file_name_path = 'Images/Test/Sonu/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        #Put count on images and display live count
        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,100), 2)
        cv2.imshow('Face Extractor', face)

    else:
        print('Face not found')
        pass

    if cv2.waitKey(1) == 13 or count == 50:
        break

cap.release()
cv2.destroyAllWindows()
print('Collecting samples completed!!!')


