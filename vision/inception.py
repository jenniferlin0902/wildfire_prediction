import argparse
import logging
import os
import random

#from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
from model.model_fn import model_fn
from model.training import train_and_evaluate
from model.utils import is_fire

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from time import time

# for trial and error
TESTING = True 
NUM_SMALL = 50

EPOCHS = 20
BATCH_SIZE = 128
LOG_FILE = 'inception_log_v1.csv'

def inception(train_data, train_labels, eval_data, eval_labels, test_data, test_labels):
	base_model = InceptionV3(weights='imagenet', include_top=False)
	x = base_model.output
	x = GlobalAveragePooling2D()(x)

	x = Dense(1024, activation='relu')(x)
	predictions = Dense(1, activation='sigmoid')(x)

	model = Model(inputs=base_model.input, outputs=predictions)

	for i, layer in enumerate(base_model.layers):
	   print(i, layer.name)

	for layer in model.layers[:249]:
		layer.trainable = False
	for layer in model.layers[249:]:
		layer.trainable = True

	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])


	print train_data.shape
	print train_labels.shape

	csv_logger = CSVLogger(LOG_FILE, append=True, separator=';')
	tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
	temp = model.fit(train_data, train_labels, validation_data=(eval_data, eval_labels), epochs=EPOCHS, 
					batch_size=BATCH_SIZE, callbacks=[csv_logger, tensorboard])
	print temp

	score = model.evaluate(test_data, test_labels, batch_size=BATCH_SIZE)
	print ("loss : " + str(score[0]))
	print ("test accuracy : " + str(score[1]))

if __name__ == '__main__':
	data_dir = 'data'
	train_data_dir = os.path.join(data_dir, "train_images")
	dev_data_dir = os.path.join(data_dir, "dev_images")
	test_data_dir = os.path.join(data_dir, "test_images")

	train_filenames = [os.path.join(train_data_dir, f.strip("_rgb.jpg")) for f in os.listdir(train_data_dir)
						if f.endswith('_rgb.jpg')]
	eval_filenames = [os.path.join(dev_data_dir, f.strip("_rgb.jpg")) for f in os.listdir(dev_data_dir)
						if f.endswith('_rgb.jpg')]
	test_filenames = [os.path.join(test_data_dir, f.strip("_rgb.jpg")) for f in os.listdir(test_data_dir)
					if f.endswith('_rgb.jpg')]


	train_labels = [is_fire(os.path.basename(f)) for f in train_filenames]
	eval_labels = [is_fire(os.path.basename(f)) for f in eval_filenames]
	test_labels = [is_fire(os.path.basename(f)) for f in test_filenames]

	train_data = []
	eval_data = []
	test_data = []

	counter = 0
	for img_path in train_filenames:
		img = image.load_img(img_path+'_ir.jpg', target_size=(299, 299))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		train_data.append(x)
		counter +=1
		if (TESTING == True and counter >=NUM_SMALL ):
			break
	counter = 0
	for img_path in eval_filenames:
		img = image.load_img(img_path+'_ir.jpg', target_size=(299, 299))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		eval_data.append(x)
		counter +=1
		if (TESTING == True and counter >=NUM_SMALL):
			break

	counter = 0
	for img_path in test_filenames:
		img = image.load_img(img_path+'_ir.jpg', target_size=(299, 299))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		test_data.append(x)
		counter +=1
		if (TESTING == True and counter >=NUM_SMALL):
			break

	train_data = np.asarray(train_data)[:,0,:,:,:]
	eval_data = np.asarray(eval_data)[:,0,:,:,:]
	test_data = np.asarray(test_data)[:,0,:,:,:]

	if (TESTING):
		inception(train_data, np.asarray(train_labels)[:NUM_SMALL], eval_data, np.asarray(eval_labels)[:NUM_SMALL],
		 test_data, np.asarray(test_labels)[:NUM_SMALL])
	else:
		inception(train_data, np.asarray(train_labels), eval_data, np.asarray(eval_labels),
		 test_data, np.asarray(test_labels))

