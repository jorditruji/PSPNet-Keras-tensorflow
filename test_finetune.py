from __future__ import print_function
from __future__ import division
from os.path import splitext, join, isfile
from os import environ
from math import ceil
import argparse
import numpy as np
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
import layers_builder as layers
import utils
import matplotlib.pyplot as plt
import os
from keras.layers import Activation, Dense, Flatten
from keras.models import Model
from pspnet import PSPNet50
from data_load import *




pspnet_ini = PSPNet50(nb_classes=150, input_shape=(640, 480),
                              weights='pspnet50_ade20k')

pspnet_ini.model.layers.pop()
last_layer = pspnet_ini.model.layers[-1].output
out = Flatten()(last_layer)
new_layer = Dense(16, activation='softmax', name='my_dense')

inp = pspnet_ini.model.input
out = new_layer(pspnet_ini.model.layers[-1].output)

model2 = Model(inp, out)

model2.compile(loss="sparse_categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
model2.summary(line_length=150)


x_train, y_train = load_data('/imatge/jmorera/PSPNet-Keras-tensorflow/train.txt', 100)
x_test, y_test = load_data('/imatge/jmorera/PSPNet-Keras-tensorflow/val.txt', 100)

print (np.squeeze(x_train).shape)
x_train= np.squeeze(x_train)
x_test = np.squeeze(x_test)

y_train=y_train.reshape(y_train.shape + (1,))
y_test=y_test.reshape(y_test.shape + (1,))
#y_train = y_train.reshape(100, 307200)
#y_test = y_test.reshape(-2, y_test.shape[-2])
print (y_train.shape)
print (y_train.shape)

model2.fit(x_train, y_train,
          batch_size=32,
          epochs=100,
          shuffle=True,
          verbose=1,
          validation_data=(x_test, y_test),
          )

'''
a=0
for layer in pspnet_ini.model.layers:
	print (layer)
	layer.trainable = False
	a=a+1
print (a)



inp = pspnet_ini.model.input                                           # input placeholder
outputs = [layer.output for layer in pspnet_ini.model.layers]          # all layer outputs
functors = [K.function([inp]+ [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

# Testing
input_shape=(640,480,3)
test = np.random.random(input_shape)[np.newaxis,...]
layer_outs = [func([test, 1.]) for func in functors]
print(layer_outs)


last = pspnet_ini.model.output
x = Flatten()(last)
x = Dense(1024, activation='relu')(x)
preds = Dense(16, activation='softmax')(x)

model = Model(pspnet_ini.input, preds)

pspnet.model.layers.pop()
pspnet.model.outputs = [pspnet.model.layers[-1].output]
pspnet.model.layers[-1].outbound_nodes = []
pspnet.model.add(Dense(16, activation='softmax'))
'''