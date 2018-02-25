import numpy as np
from datetime import datetime as dt
from PIL import Image, ImageDraw
import json
import os
from pprint import pprint

import scipy
from scipy import ndimage
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from scipy import misc

FIRE_LABEL_FILE='data/landsat_fire_last7.csv'
BASE = "/Volumes/DRIVE/"
DEFAULT_DOWNLOAD_PATH=os.path.join(BASE,'landsat_download')
OUPUT_PATH=os.path.join(BASE,"landsat_cropped")

# sorry, hardcoded & hopefully only called once (angle = 14) 
def find_rotation_angle(data):
    x_values = data[:,0,0]
    y_values = data[0,:,0]

    xpad = -1
    ypad = -1

    for i in range(len(x_values)):
        for j in range(len(y_values)):
            if (data[i, j, 0] != 0):
                print 'value: ' + str(data[i, j, 0]) + ' index: ' + str(i) + ', ' + str(j)
                ypad = j
                break
        if (ypad != -1):
            break

    for j in range(len(y_values)):
        for i in range(len(x_values)):
            if (data[i, j, 0] != 0):
                print 'second value: ' + str(data[i, j, 0]) + ' index: ' + str(i) + ', ' + str(j)
                xpad = i
                break
        if (xpad != -1):
            break
    # got results: value: 244 index: 14, 1583
    # second value: 20 index: 6351, 140; so 1583/6351


# changed to directly pass a pillow image
def straighten_image(im):
    # 13.99 found using find_rotation_angle()
    im = im.rotate(13.99)
    box = im.getbbox()
    im = im.crop(box)
    return im

def find_corner_coords(filename):
    f = open(filename,'r')
    lines = f.readlines()
    lines = [line for line in lines if line.strip()]
    for i, l in enumerate(lines):
        if (l.find('CORNER_UL_LAT_PRODUCT') != -1):
            ul_lat = float(l.split('=')[1].strip())
            # there might have been a ` in the file, in which case, need to replace it... w/ .replace('`', '')
            ul_lon = float(lines[i+1].split('=')[1].strip())
            ur_lat = float(lines[i+2].split('=')[1].strip())
            ur_lon = float(lines[i+3].split('=')[1].strip())
            ll_lat = float(lines[i+4].split('=')[1].strip())
            ll_lon = float(lines[i+5].split('=')[1].strip())
            lr_lat = float(lines[i+6].split('=')[1].strip())
            lr_lon = float(lines[i+7].split('=')[1].strip())
            break

    coords = {'ul_lat': ul_lat, 'ul_lon': ul_lon, 'ur_lat': ur_lat, 'ur_lon': ur_lon,
              'll_lat': ll_lat, 'll_lon': ll_lon, 'lr_lat': lr_lat, 'lr_lon': lr_lon}

    return coords

def map_fires(image, coords, filename, label, mini_size):
    im_data = np.asarray(image)
    num_across = min(im_data.shape[0], im_data.shape[1])/mini_size
    is_fire = np.zeros((num_across * num_across))

    with open(filename) as f:
        data = json.load(f)

    for i in range(len(data[unicode(label)]['lats'])):
   
        lat = data[label]['lats'][i]
        lon = data[label]['lons'][i]

        lat_range = abs(coords['ul_lat'] - coords['ll_lat'])
        lon_range = abs(coords['ur_lon'] - coords['ul_lon'])


        lat_index = int((lat - coords['ll_lat'])/(lat_range/num_across))
        lon_index = int((lon - coords['ul_lon'])/(lon_range/num_across))


        if (lat_index < num_across and lon_index < num_across):
            is_fire[lon_index*num_across + lat_index] = 1
            print str(lon_index*num_across + lat_index)

    return is_fire

def image_blocks(image, label, is_fire, mini_size):
    data = np.asarray(image)
    size = (mini_size, mini_size)

    # number of mini images that can fit in one dimension
    num_across = min(data.shape[0], data.shape[1])/mini_size

    # crop dimensions (left, upper, right, lower)
    for i in range(num_across):
        for j in range(num_across):
            temp = image.crop((i * mini_size, j * mini_size, (i+1) * mini_size, (j+1) * mini_size))
            # fire_label: 0 for no fire, 1 for fire
            outfile_name = label + '_' + str(i*num_across + j) + '_' + str(int(is_fire[i*num_across + j])) + '.jpg'
            print outfile_name
            outfile_name = os.path.join(OUPUT_PATH, outfile_name)
            temp.save(outfile_name, "JPEG")

# outputs .npy file instead of jpeg; to open use np.load('')
def image_blocks_with_ir(image, ir_image, label, is_fire, mini_size):
    data = np.asarray(image)
    size = (mini_size, mini_size)
    # number of mini images that can fit in one dimension
    num_across = min(data.shape[0], data.shape[1])/mini_size

    # crop dimensions (left, upper, right, lower)
    for i in range(num_across):
        for j in range(num_across):
            temp = image.crop((i * mini_size, j * mini_size, (i+1) * mini_size, (j+1) * mini_size))
            temp_ir = ir_image.crop((i * mini_size, j * mini_size, (i+1) * mini_size, (j+1) * mini_size))

            temp_data = np.asarray(temp)
            temp_ir_data = np.asarray(temp_ir)

            # fire_label: 0 for no fire, 1 for fire
            outfile_name = label + '_' + str(i*num_across + j) + '_' + str(int(is_fire[i*num_across + j]))
            print outfile_name

            final_array = np.concatenate((temp_data, temp_ir_data.reshape(mini_size, mini_size, 1)), axis = 2)
            outfile_name = os.path.join(OUPUT_PATH, outfile_name)
            np.save(outfile_name, final_array)


def plot_fires(irimage, coords, filename):
    # latidue goes up down, index 0
    # longitude = across, index 0

    lat_range = abs(coords['ul_lat'] - coords['ll_lat'])
    lon_range = abs(coords['ur_lon'] - coords['ul_lon'])

    im_data = np.asarray(irimage)
    plt.imshow(im_data, cmap='gray')

    with open(filename) as f:
        data = json.load(f)

    for i in range(len(data[unicode(label)])):
        lat = data[label]['lats'][i]
        lon = data[label]['lons'][i]
        print lat
        print lon
        # when plotting coordinates, x first, then y (y also no longer defined from top)
        plt.scatter((float(lon)-coords['ul_lon'])*im_data.shape[0]/lon_range, im_data.shape[1]-(float(lat)-coords['ll_lat'])*im_data.shape[1]/lat_range, s=5)
    plt.show()

if __name__ == '__main__':

    #image_list_path = os.path.join(FIRE_LABEL_FILE)
    #print "opening {}".format(FIRE_LABEL_FILE)

    #image_list = [
    #	"LC08_L1TP_032026_20180221_20180221_01_RT",
#		"LC08_L1TP_030026_20180223_20180223_01_RT",
#		"LC08_L1TP_030025_20180223_20180223_01_RT",
#		"LC08_L1TP_036026_20180217_20180217_01_RT"
#	]
    image_list_path = os.path.join(DEFAULT_DOWNLOAD_PATH, "landsat_fire_2017_scene_lists.json")
    meta_list_path = os.path.join(DEFAULT_DOWNLOAD_PATH, "landsat_fire_2017_meta.json")
    with open(image_list_path, 'r') as image_list_f:
        image_list = json.load(image_list_f)
    mini_size = 300
    images_base = DEFAULT_DOWNLOAD_PATH
    for label in image_list:
        # remove the label
        #label = 'LC08_L1TP_027033_20180218_20180218_01_RT'
        #label = 'LC08_L1TP_045033_20180216_20180216_01_RT'
        print "Cropping image {}".format(label)
        image_dir = os.path.join(images_base, label)
        irimage_input = os.path.join(image_dir, label + '_B7.TIF')
        processedimage_input = os.path.join(os.path.join(image_dir, label), label + '_bands_234.TIF')
        coordfile = os.path.join(image_dir, label + '_MTL.txt')
        if not (os.path.exists(image_dir) and os.path.exists(processedimage_input) and os.path.exists(coordfile)):
            print "Image directory or file {} does not exist".format(image_dir)
            continue

        im = Image.open(processedimage_input)
        image = straighten_image(im)
        coords = find_corner_coords(coordfile)

        ir_im = cv2.imread(irimage_input, 0)
        ir_image = straighten_image(Image.fromarray(ir_im))

        #plot_fires(ir_pillow_image, coords, firefile)

        # returns an array: index corresponds to label of mini image
        # 0 for no fire, 1 for fire
        is_fire = map_fires(image, coords, meta_list_path, label, mini_size)

        #image_blocks(image, label, is_fire, mini_size)
        image_blocks_with_ir(image, ir_image, label, is_fire, mini_size)

        # remove the break'''
        #break
