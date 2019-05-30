# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 14:20:40 2018

@author: varinder
"""

from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.datasets import mnist
from keras.utils import to_categorical
import pandas as pd

classifier = Sequential()
classifier.add(Convolution2D(64, 3, 3, input_shape = (45, 45, 1), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units= 128, activation = 'relu'))

classifier.add(Dense(units=34, activation='softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('train_data/train',
                                                 target_size = (45, 45),color_mode='grayscale'
                                                , batch_size = 64,
                                                 )
test_datagen = ImageDataGenerator(rescale = 1./255)


test_set = test_datagen.flow_from_directory('train_data/test',
                                           color_mode='grayscale' ,target_size = (45, 45),
                                            batch_size = 64)

valid_datagen = ImageDataGenerator(rescale = 1./255)


valid_set = valid_datagen.flow_from_directory('train_data/validation',
                                           color_mode='grayscale' ,target_size = (45, 45),
                                            batch_size = 64)
classifier.fit_generator(training_set,
                         steps_per_epoch = 128,
                         nb_epoch = 20,validation_data = valid_set
                                   )

classifier.save('modelfinalvarpc.h5')
classifier.evaluate_generator(test_set)






from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('classes',
                                                 target_size = (45, 45),color_mode='grayscale'
                                                , batch_size = 64,
                                                 )


from keras.models import load_model
model = load_model('modelfinalvarpc.h5')
'''model.evaluate_generator(test_set)'''

oi=training_set.class_indices

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img=Image.open('five.jpg').convert('L') 
img=np.asarray(img)
img=img.reshape(-1,45,45,1)
'''img=img.astype('float64')
img=img/255'''
hello=model.predict(img)
p=np.argmax(hello,axis=1)
for sign,val in oi.items():
    if val==p:
        s=sign


if(s=='div'):
    s='/'
elif(s=='times'):
    s='*'

print(s)

