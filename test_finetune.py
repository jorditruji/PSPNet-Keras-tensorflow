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

from pspnet import PSPNet50




pspnet_ini = PSPNet50(nb_classes=16, input_shape=(640, 480),
                              weights='pspnet50_ade20k')


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