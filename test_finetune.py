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
from keras.layers import Activation, Dense, Flatten, Conv2D, Lambda, Input, Reshape
from keras.models import Model,Sequential
from pspnet import PSPNet50
from data_load import *


def resize_like(input_tensor, ref_tensor): # resizes input tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value, W.value])




pspnet_ini = PSPNet50(nb_classes=150, input_shape=(640, 480),
                              weights='pspnet50_ade20k')

pspnet_ini.model.summary(line_length=150)
pspnet_ini.model.layers.pop()
layer_lambda = pspnet_ini.model.layers.pop()
pspnet_ini.model.layers.pop()
print(layer_lambda.get_config())
'''
layer_lambda = pspnet_ini.model.layers.pop()
layer_lambda.get_config()
'''

tf_resize = Input(shape=(640, 480))
kernel_size=(1,1)
new_layer=Conv2D(16, (1, 1), strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='linear', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(pspnet_ini.model.layers[-1].output)
#new_layer = Dense((640, 480), activation='relu', name='fc2')(new_layer)
new_layer = Lambda(resize_like, arguments={'ref_tensor':tf_resize},name='custom')(new_layer)
inp = pspnet_ini.model.input
out =Dense(16, activation='softmax', name='my_dense')(new_layer)
out = Reshape((640*480, num_classes))(out)
model2 = Model(inp, out)



'''
out =Dense(16, activation='softmax', name='my_dense')(new_layer)


#{"name": "conv6", "config": {"filters": 16, "use_bias": true, "name": "conv6", "bias_regularizer": null, "strides": [1, 1], "data_format": "", "activation": "linear", "trainable": true, "kernel_constraint": null, "activity_regularizer": null, "padding": "valid", "bias_initializer": {"config": {}, "class_name": "Zeros"}, "kernel_initializer": {"config": {"seed": null, "distribution": "uniform", "scale": 1.0, "mode": "fan_avg"}, "class_name": "VarianceScaling"}, "bias_constraint": null, "kernel_size": [1, 1], "kernel_regularizer": null, "dilation_rate": [1, 1]}, "class_name": "Conv2D"
last_layer = pspnet_ini.model.layers[-1].output
#new_layer = Flatten()(last_layer)

new_layer = Dense((640, 480), activation='relu', name='fc2')(new_layer)

out =Dense(16, activation='softmax', name='my_dense')(new_layer)


#out = new_layer(pspnet_ini.model.layers[-1].output)

inp = pspnet_ini.model.input
model2 = Model(inp, out)

#model2.summary(line_length=150)
'''
model2.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])
x_train, y_train = load_data('/imatge/jmorera/PSPNet-Keras-tensorflow/train.txt', 100)
x_test, y_test = load_data('/imatge/jmorera/PSPNet-Keras-tensorflow/val.txt', 100)

print (np.squeeze(x_train).shape)
x_train= np.squeeze(x_train)
x_test = np.squeeze(x_test)


#y_train = y_train.reshape(100, 307200)
#y_test = y_test.reshape(-2, y_test.shape[-2])
print (y_train.shape)
print (y_test.shape)

a=0
for layer in model2.layers[:222]:
    layer.trainable = False


model2.fit(x_train, y_train,
          batch_size=6,
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