import numpy as np
import glob, os
from PIL import Image
import os

BASE = "vision/data/dev_images"
NPY_PATH=BASE
OUTPUT_PATH=BASE

rotate = False

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
		for i in range(1,4):
			rotated_file_name=os.path.join(OUTPUT_PATH, "{}-".format(i) + os.path.basename(outfile_name))
			im = im.rotate(90)
			im.save(rotated_file_name)


# alternative:
#os.chdir(NPY_PATH)
#for file in glob.glob("*.npy"):
none_fire_files = [f for f in os.listdir(NPY_PATH) if f.endswith(".npy") and not is_fire(f)]
keep_count = len(none_fire_files)/3
print "Found {} nono_fire_files, keep {}".format(len(none_fire_files), keep_count)
toss_files = none_fire_files[keep_count:]
print len(toss_files)
print toss_files

for file in os.listdir(NPY_PATH):
	if file.endswith(".npy"):
		if file in toss_files:
			os.remove(os.path.join(NPY_PATH, file))
		else:
			convert_file(file)