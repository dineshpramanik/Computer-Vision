from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input    #Transfer learning technique
from keras.applications.vgg19 import VGG19, preprocess_input    #Transfer learning technique
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

#resize all the images
img_size = [224,224]

train_path = 'Images/Train'
validation_path = 'Images/Test'



#Preprocessing layer to the front of VGG16
#vgg16 = VGG16(input_shape = img_size + [3], weights='imagenet', include_top=False)


#Preprocessing layer to the front of VGG19
vgg19 = VGG19(input_shape = img_size + [3], weights='imagenet', include_top=False)



#Exclude training of  existing layers
#for layer in vgg16.layers:
for layer in vgg19.layers:
    layer.trainable = False

#Number of classes
folder = glob('Images/Train/*')

#Output layers
#x= Flatten()(vgg16.output)
x= Flatten()(vgg19.output)
prediction = Dense(len(folder), activation='softmax')(x)

#Create a model object
#model = Model(inputs=vgg16.input, outputs= prediction)
model = Model(inputs=vgg19.input, outputs= prediction)

#View the structure of model
model.summary()

#Model optimization
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('Images/Train/',
                                                 target_size=(224,224),
                                                 batch_size=8,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('Images/Test/',
                                                 target_size=(224,224),
                                                 batch_size=8,
                                                 class_mode='categorical')

#Fit the model
r= model.fit_generator(train_set,
                       validation_data=test_set,
                       epochs=5,
                       steps_per_epoch=len(train_set),
                       validation_steps=len(test_set))

#Loss
plt.plot(r.history['loss'], label='Train loss')
plt.plot(r.history['val_loss'], label='Validation loss')

#Accuracy
plt.plot(r.history['accuracy'], label='Train accuracy')
plt.plot(r.history['val_accuracy'], label='Validation accuracy')

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.figure()

#plt.savefig('face_recognition_vgg16.png')
plt.savefig('face_recognition_vgg19.png')

#model.save('facerecognition_model_vgg16.h5')
model.save('facerecognition_model_vgg19.h5')