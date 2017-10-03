from __future__ import print_function
from __future__ import division
from os.path import splitext, join, isfile
from os import environ
from math import ceil
from keras.optimizers import SGD
import argparse
from keras.utils import np_utils
import numpy as np
from scipy import misc, ndimage
from keras import backend as K
from keras.models import model_from_json
import tensorflow as tf
import layers_builder as layers
import utils
import matplotlib


def class_weighted_pixelwise_crossentropy(target, output):
     output = tf.clip_by_value(output, 10e-8, 1.-10e-8)

     #with open('class_weights.pickle', 'rb') as f:
     weight = [0.005, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 ,0.1 ,0. ,0.1 ,0.1 ,0.1]
     return -tf.reduce_sum(target * weight * tf.log(output))

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import os
from keras.layers import Activation, Dense, Flatten, Conv2D, Lambda, Input, Reshape
from keras.models import Model,Sequential
from pspnet import PSPNet50
from data_load import *

def depth_softmax(matrix):
    sigmoid = lambda x: 1 / (1 + K.exp(-x))
    sigmoided_matrix = sigmoid(matrix)
    softmax_matrix = sigmoided_matrix / K.sum(sigmoided_matrix, axis=0)
    return softmax_matrix

def resize_like(input_tensor, ref_tensor): # resizes input tensor wrt. ref_tensor
    H, W = ref_tensor.get_shape()[1], ref_tensor.get_shape()[2]
    return tf.image.resize_nearest_neighbor(input_tensor, [H.value, W.value])






def plot_metrics(history):

    print(history.history.keys())

    fig = matplotlib.pyplot.figure(1)

    # summarize history for accuracy

    matplotlib.pyplot.subplot(211)
    matplotlib.pyplot.plot(history.history['acc'])
    matplotlib.pyplot.plot(history.history['val_acc'])
    matplotlib.pyplot.title('model accuracy')
    matplotlib.pyplot.ylabel('accuracy')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    matplotlib.pyplot.subplot(212)
    matplotlib.pyplot.plot(history.history['loss'])
    matplotlib.pyplot.plot(history.history['val_loss'])
    matplotlib.pyplot.title('model loss')
    matplotlib.pyplot.ylabel('loss')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.legend(['train', 'test'], loc='upper left')
    fig.savefig('metrics.png', dpi=fig.dpi)

# Load dataset

pspnet_ini = PSPNet50(nb_classes=150, input_shape=(640, 480),
                              weights='pspnet50_ade20k')


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
#
out = Reshape((640*480, 16))(out)
#out =Flatten()(out)

#out=Lambda(depth_softmax, name='custom2')(new_layer)
#out=Reshape((640*480, 16))(out)
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

x_train, y_train = load_data('/imatge/jmorera/PSPNet-Keras-tensorflow/train.txt', 1000)
x_test, y_test = load_data('/imatge/jmorera/PSPNet-Keras-tensorflow/val.txt', 450)

x_train= np.squeeze(x_train)
x_test = np.squeeze(x_test)
list_y_train =[]
list_y_test=[]
#y_train = y_train.reshape(100, 307200)
#y_test = y_test.reshape(100, 307200)


'''
y_train = np_utils.to_categorical(y_train, 16)
y_test = np_utils.to_categorical(y_test, 16)

y_train=y_train.reshape((100, 640 * 480 * 16))
y_test=y_test.reshape((100, 640 * 480 * 16))
a=0
'''
for layer in model2.layers[:-8]:
    layer.trainable = False
sgd = SGD(lr=0.001, momentum=0, decay=0.002, nesterov=True)
model2.compile(loss=class_weighted_pixelwise_crossentropy, optimizer=sgd, metrics=['accuracy'])

model2.summary(line_length=150)


print (y_train.shape)

history=model2.fit(x_train, y_train,
          batch_size=8,
          epochs=50,
          shuffle=True,
          verbose=1,
          validation_data=(x_test, y_test),
          )
plot_metrics(history)
#predict_labels=model.predict(data.X_val)


#for prediction,orig_prediction in zip(predict_labels,data.labels_val):
  #  ind1 = np.argmax(prediction)
   # ind2= np.argmax(orig_prediction)
   # predictions.append(ind1)
   # orig_predictions.append(ind2)

#cm=confusion_matrix(orig_predictions, predictions)

#plot_confusion_matrix(cm, data.names_class)

