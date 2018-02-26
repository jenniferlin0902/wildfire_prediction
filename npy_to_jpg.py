import numpy as np
import glob, os
from PIL import Image
import os

BASE = "vision/data/train_images"
NPY_PATH=BASE
OUTPUT_PATH=BASE

rotate = True

def is_fire(filename):
    return int(filename.split(".")[0][-1])

def convert_file(file):
	data = np.load(os.path.join(NPY_PATH,file))
	data = data[:,:,:3]
	im = Image.fromarray(data)
	outfile_name = os.path.join(OUTPUT_PATH, str(file[:-4]) + '.jpg')
	im.save(outfile_name, "JPEG")
	# -4 to remove '.npy'
	if rotate and is_fire(file):
		for i in range(3):
			rotated_file_name=os.path.join(OUTPUT_PATH, "{}-".format(i) + os.path.basename(outfile_name))
			im = im.rotate(90)
			im.save(rotated_file_name)



# alternative:
#os.chdir(NPY_PATH)
#for file in glob.glob("*.npy"):

for file in os.listdir(NPY_PATH):
	if file.endswith(".npy"):
		convert_file(file)
		exit(1)