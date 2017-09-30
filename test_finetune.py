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

from pspnet import PSPNet50




pspnet = PSPNet50(nb_classes=16, input_shape=(640, 480),
                              weights='pspnet50_ade20k')

pspnet.model.layers.pop()
pspnet.model.outputs = [pspnet.model.layers[-1].output]
pspnet.model.layers[-1].outbound_nodes = []
pspnet.model.add(Dense(16, activation='softmax'))
