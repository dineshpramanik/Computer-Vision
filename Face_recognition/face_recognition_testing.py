# Importing the libraries
from PIL import Image
import cv2
import numpy as np
from keras.models import load_model

#Loading model
model = load_model('facerecognition_model_vgg19.h5')

#Loading HAAR face classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    faces = face_cascade.detectMultiScale(img,1.3, 5)

    if faces is ():
        return None

    #Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        y = y - 10
        cropped_face = img[y:y + h + 20, x:x + w + 20]

    return cropped_face

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    face = face_extractor(frame)

    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))

        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)

        # Our keras model used a 4D tensor, (images x height x width x channel)
        # So changing dimension 128x128x3 into 1x128x128x3

        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        #print(pred)

        name = "None matching"

        if (pred[0][0]>0.5):
            name = 'Dinesh'

        elif (pred[0][1]>0.5):
            name = 'Drishty'

        cv2.putText(frame, name, (20,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)

    else:
        cv2.putText(frame, 'No face found', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()