import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


#mobile = keras.applications.mobilenet.MobileNet()

#########################################
from keras.models import Sequential
# To save model
#model.save(r"C:\Users\Asus\Desktop\my_model_01.hdf5')

# To load the model

# To load a persisted model that uses the CRF layer 
model1 = keras.models.load_model(r"C:\Users\Asus\Desktop\my_model_01.hdf5")
########################################

model = Sequential()

for layer in model1.layers[:-5]:
    model.add(layer)


model.add(Dense(1024,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(3,activation='softmax'))

#model.build()
#model.summary()

for layer in model.layers[:90]:
    layer.trainable=False


########################################
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory(r'C:\Users\Asus\Desktop\train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)


model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=5)



