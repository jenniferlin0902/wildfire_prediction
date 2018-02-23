import numpy as np
from datetime import datetime as dt
from PIL import Image
import json
import os
from pprint import pprint

FIRE_LABEL_FILE='data/landsat_fire_last7.csv'
DEFAULT_DOWNLOAD_PATH='landsat_download'

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
    #got results: value: 244 index: 14, 1583
    # second value: 20 index: 6351, 140; so 1583/6351

def straighten_image(imagename):
    im= Image.open(imagename)
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
            lr_lon = float(lines[i+6].split('=')[1].strip())
            break

    coords = {'ul_lat': ul_lat, 'ul_lon': ul_lon, 'ur_lat': ur_lat, 'ur_lon': ur_lon,
              'll_lat': ll_lat, 'll_lon': ll_lon, 'lr_lat': lr_lat, 'lr_lon': lr_lon}

    return coords

def map_fires(image, coords, filename, label, mini_size):
    data = np.asarray(image)
    num_across = min(data.shape[0], data.shape[1])/mini_size
    is_fire = np.zeros((num_across, num_across))

    with open(filename) as f:
        data = json.load(f)

    for i in range(len(data[label])):
        print data[label]['lat'][i]
        lat = data[label]['lat'][i]
        lon = data[label][i]['lon'][i]
        lat_index = (lat - coords['ll_lat'])/mini_size
        lon_index = (lon - coords['ul_lon'])/mini_size

        if (lat_index < num_across and lon_index < num_across):
            is_fire[lat_index*num_across + lon_index] = 1
    return is_fire

def image_blocks(image, label, is_fire, mini_size):
    data = np.asarray(image)
    mini_size = is_fire.shape[0]
    size = (mini_size, mini_size)

    # number of mini images that can fit in one dimension
    num_across = min(data.shape[0], data.shape[1])/mini_size

    for i in range(num_across):
        for j in range(num_across):
            temp = image.crop((i * mini_size, (i+1) * mini_size, j * mini_size, (j+1) * mini_size))
            #fire_label = get_fire_label()
            # TODO is this use ????
            # fire_label: 0 for no fire, 1 for fire
            outfile_name = label + '_' + str(i*num_across + j) + is_fire[i*num_across + j] + '.jpg'
            temp.save(outfile_name, "JPEG")

if __name__ == '__main__':

    image_list_path = os.path.join(DEFAULT_DOWNLOAD_PATH, 'landsat_fire_last7_meta.json')
    images_base = DEFAULT_DOWNLOAD_PATH
    with open(image_list_path, 'r') as image_list_f:
        image_list = json.load(image_list_f)
    mini_size = 300

    for label in image_list:
        label = 'LC08_L1TP_026032_20180211_20180211_01_RT'
        image_dir = os.path.join(images_base, label)
        irimage = os.path.join(image_dir, label + '_B7.TIF')
        processedimage = os.path.join(os.path.join(image_dir, label), label + '_bands_234.TIF')
        coordfile = os.path.join(image_dir, label + '_MTL.txt')
        firefile = os.path.join(DEFAULT_DOWNLOAD_PATH,'landsat_fire_last7_meta.json')
        image = straighten_image(processedimage)
        coords = find_corner_coords(coordfile)

        # returns an array: index corresponds to label of mini image
        # 0 for no fire, 1 for fire
        is_fire = map_fires(image, coords, firefile, label, mini_size)

        image_blocks(image, label, is_fire)
