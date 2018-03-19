import argparse
import logging
import os
import random

#from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.utils import save_dict_to_json
#from model.model_fn import model_fn
from model.training import train_and_evaluate
from model.utils import is_fire

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import numpy as np
from keras.callbacks import CSVLogger
#from keras.callbacks import TensorBoard
from time import time

# for trial and error
TESTING = False
NUM_SMALL = 12
# make sure this is > BATTCH_SIZE or else error in validation_steps -> 0

EPOCHS = 20
BATCH_SIZE = 5
LOG_FILE = 'inception_log_v2.csv'

# from https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
def generator(file_list, batch_size):
	#batch_features = np.zeros((batch_size, 299, 299, 3))
	#batch_labels = np.zeros((batch_size,1))
	i = 0
	while True:
		image_batch = np.zeros((batch_size, 299, 299, 3))
		label_batch = np.zeros((batch_size,1))
		for b in range(batch_size):
			if (i == (len(file_list))):
				i = 0
				random.shuffle(file_list)
			filename = file_list[i]
			img = image.load_img(filename+'_ir.jpg', target_size=(299, 299))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			image_batch[b] = x
			label_batch[b] = is_fire(os.path.basename(filename))
			i += 1
		yield image_batch, label_batch


# note that pasing in the labels correctly, but now, 
# need to load images on the fly
def inception(train_data_list, eval_data_list, test_data_list):
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


	csv_logger = CSVLogger(LOG_FILE, append=True, separator=';')
	#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
	#temp = model.fit(train_data, train_labels, validation_data=(eval_data, eval_labels), epochs=EPOCHS, 
					#batch_size=BATCH_SIZE, callbacks=[csv_logger])

	temp = model.fit_generator(generator(test_filenames, BATCH_SIZE), steps_per_epoch=len(test_filenames)//BATCH_SIZE, 
    				validation_data = generator(eval_filenames, BATCH_SIZE), validation_steps = len(eval_filenames)//BATCH_SIZE,
                    epochs = 20, 
                    callbacks=[csv_logger])
	print temp

	#score = model.evaluate(test_data, test_labels, batch_size=BATCH_SIZE)
	score = model.evaluate_generator(generator(test_filenames, BATCH_SIZE), steps=len(test_filenames)//BATCH_SIZE)
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

	if (TESTING == True):
		train_filenames = train_filenames[:NUM_SMALL]
		eval_filenames = eval_filenames[:NUM_SMALL]
		test_filenames = test_filenames[:NUM_SMALL]

	inception(train_filenames, eval_filenames, test_filenames)

