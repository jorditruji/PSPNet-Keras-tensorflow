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

import utils
import matplotlib
import cv2

from data_load import *


foto = read_label('/projects/world3d/2017-06-scannet/scene0405_00/frame-001850.depth.pgm.mat')
misc.imsave('outfile.jpg', foto)