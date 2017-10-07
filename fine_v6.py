from __future__ import print_function
from __future__ import division
from os.path import splitext, join, isfile
from os import environ
from math import ceil
from keras.optimizers import SGD, Adam
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
import cv2
import scipy.io

def w_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        nb_cl = len(weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1, keepdims=True)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
        return K.categorical_crossentropy(y_pred, y_true) * final_mask
    return loss

def class_weighted_pixelwise_crossentropy(target, output):
    output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
    #with open('class_weights.pickle', 'rb') as weights:
    weights = [0.0, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5 ,1e-5 ,1e-5 ,1e-5 ,1e-5 ,1e-5]
    return -tf.reduce_sum(target * weights * tf.log(output))





     #weight = [0.0, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5 ,1e-5 ,1e-5 ,1e-5 ,1e-5 ,1e-5]

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

def img2int(img):
    zmax=np.max(img)
    norm_img=np.zeros(img.shape,dtype=np.uint8)
    mask=np.zeros(img.shape,dtype=np.uint8)
    mask_std=np.zeros(img.shape,dtype=np.uint8)
    cont=0
    for pos in product(range(h), range(w)):
    #for idx in img:
        pixel =  img.item(pos[0],pos[1])
        if pixel>0:
            new_pix=np.multiply((((1.0/float(pixel))-(1.0/float(zmax)))/((1.0-(1.0/float(zmax))))),1.0)
            new_pix2=(float(pixel)/float(zmax))*254.0
           # print new_pix
      #  print new_pix
            norm_img[pos]=255-new_pix2
            mask[pos]=0
            mask_std[pos]=255
        else:
            norm_img[pos]=0
            if (pos[1]>20 and pos [0]>20):
                mask[pos]=255
                mask_std[pos]=0
            else:
                mask[pos]=255
                mask_std[pos]=0
        cont+=1
    #print (np.unique(norm_img))

    dst_TELEA = cv2.inpaint(norm_img,mask,3,cv2.INPAINT_TELEA)
    dst_TELEA=equalize_hist(dst_TELEA)
    return dst_TELEA

def equalize_hist(img):
    equ = cv2.equalizeHist(img)
    #res = np.hstack((img,equ)) #stacking images side-by-side
    return equ
    #cv2.imwrite('res.png',res)


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


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
    fig.savefig('/imatge/epresas/depth.png', dpi=fig.dpi)

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
new_layer=Conv2D(256, (1, 1), strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation='linear', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(pspnet_ini.model.layers[-1].output)
#new_layer = Dense((640, 480), activation='relu', name='fc2')(new_layer)
new_layer = Lambda(resize_like, arguments={'ref_tensor':tf_resize},name='custom')(new_layer)
inp = pspnet_ini.model.input
out =Dense(256, activation='softmax', name='my_dense')(new_layer)
#
#out = Reshape((640*480, 256))(out)
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



#x_train, y_train = load_data('/imatge/jmorera/PSPNet-Keras-tensorflow/train.txt', 600)
#x_test, y_test = load_data('/imatge/jmorera/PSPNet-Keras-tensorflow/val.txt', 300)

#x_train= np.squeeze(x_train)
#x_test= load_data('/imatge/jmorera/PSPNet-Keras-tensorflow/val.txt', 1)


#y_train = y_train.reshape(100, 307200)
#y_test = y_test.reshape(100, 307200)


'''
y_train = np_utils.to_categorical(y_train, 16)
y_test = np_utils.to_categorical(y_test, 16)

y_train=y_train.reshape((100, 640 * 480 * 16))
y_test=y_test.reshape((100, 640 * 480 * 16))
a=0
'''
for layer in model2.layers[:-6]:
    layer.trainable = False



sgd = SGD(lr=0.001, momentum=0, decay=0.002, nesterov=True)
adam=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

model2.compile(loss='mean_squared_logarithmic_error', optimizer=adam, metrics=['accuracy'])

model2.summary(line_length=150)


history= model2.fit_generator(
     load_data_V2('/imatge/jmorera/PSPNet-Keras-tensorflow/train.txt', 4),
      steps_per_epoch = 600,
       nb_epoch = 20,
        verbose=1,
          validation_data=load_data_V2('/imatge/jmorera/PSPNet-Keras-tensorflow/val.txt', 4),
          validation_steps=180)

#history=model2.fit(x_train, y_train,
#          batch_size=8,
 #         epochs=12,
  #        shuffle=True,
   #       verbose=1,
    #      validation_data=(x_test, y_test),
     #     )
plot_metrics(history)
model2.save_weights('pesos_pesants_light.h5')
x_test, y_test = load_data('/imatge/jmorera/PSPNet-Keras-tensorflow/test.txt', 1)
predict_labels=model2.predict(x_test)
scipy.io.savemat('out.mat', mdict={'exon': predict_labels})
prediction=tf.argmax(predict_labels,1)
scipy.io.savemat('out2.mat', mdict={'exon': prediction})

#600/600 [==============================] - 2507s - loss: 3.4009 - acc: 0.0433 - val_loss: 15.9909 - val_acc: 0.0213

#for prediction,orig_prediction in zip(predict_labels,data.labels_val):
  #  ind1 = np.argmax(prediction)
   # ind2= np.argmax(orig_prediction)
   # predictions.append(ind1)
   # orig_predictions.append(ind2)

#cm=confusion_matrix(orig_predictions, predictions)

#plot_confusion_matrix(cm, data.names_class)

