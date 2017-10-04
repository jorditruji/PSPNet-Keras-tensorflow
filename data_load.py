import subprocess
import os, sys
import numpy as np
from itertools import islice
import time
import shutil
from keras.utils import np_utils
import random
import numpy as np
from scipy import misc, ndimage, io

DATA_MEAN = np.array([[[126.92261499, 114.11585906, 99.15394194]]])  # RGB order


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '|'):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total:
		print()


def read_image(name):
	img=misc.imread(name)
	img_Resize= misc.imresize(img, (640, 480))

	return img_Resize

def read_label(name):
	label=io.loadmat(name)[name[:-4]]
	label=np.transpose(label.astype('uint8'))
	label = label.ravel()
	label = np_utils.to_categorical(label, 16)
	#print (label.shape)
	return label


def load_data(path,num_img):
	filename = path
	images =[]
	labels=[]
	cont=0
	i=0
	with open(filename) as f:
		head = list(islice(f, num_img))
		head=random.sample(f.readlines(),num_img)
		for line in head:
			printProgressBar(i + 1, len(head), prefix='Progress:', suffix='Complete', length=50)
			i += 1
			#print (line)

			prova =line.strip().split(' ')
			img=read_image(prova[0])
			float_img = img.astype('float16')
			centered_image = float_img - DATA_MEAN
			bgr_image = centered_image[:, :, ::-1]  # RGB => BGR
			input_data = bgr_image[np.newaxis, :, :, :] 
			images.append(input_data)
			labels.append(read_label(prova[1]))
	images=np.array(images)
	labels=np.array(labels)
	images= np.squeeze(images)
	print (images.shape)
	print (labels.shape)
	#labels = labels.reshape(num_img, 307200)
#y_test = y_test.reshape(100, 307200)

#y_train = y_train.reshape(100, 307200)
#y_test = y_test.reshape(100, 307200)

	return (images, labels)

def create_mean(path):
	filename = path
	images =[]
	labels=[]
	with open(filename) as f:
		for line in f:
			prova =line.strip().split(' ')

			images.append(calc_mean(read_image(prova[0])))

	print (sum(images) / float(len(images)))
	return sum(images) / float(len(images))


def calc_mean(image):

	return np.mean(image, axis=(0, 1))


