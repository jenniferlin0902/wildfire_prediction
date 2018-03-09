import os
import numpy as np
from pprint import pprint
import json
from datetime import datetime as dt
import scipy
from scipy import ndimage
from scipy import misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import utm

#BASE = '/Volumes/Transcend/'
#DEFAULT_DOWNLOAD_PATH=os.path.join(BASE,'landsat_download_3')
#OUPUT_PATH=os.path.join(BASE,'landsat_download_3_cropped')'''

### for local testing:
BASE = '/Volumes/passport'
DEFAULT_DOWNLOAD_PATH=os.path.join(BASE,'landsat_download_2017')
OUTPUT_PATH = os.path.join(BASE, 'landsat_cropped_2017_testing')

def get_metainfo(filename):
    f = open(filename,'r')
    lines = f.readlines()
    lines = [line for line in lines if line.strip()]
    for i, l in enumerate(lines):
        if (l.find('CORNER_UL_PROJECTION_X_PRODUCT') != -1):
            ul_x = float(l.split('=')[1].strip())
            ul_y = float(lines[i+1].split('=')[1].strip())
            lr_x = float(lines[i+6].split('=')[1].strip())
            lr_y= float(lines[i+7].split('=')[1].strip())
        elif (l.find('UTM_ZONE') != -1):
            utm_zone = int(l.split('=')[1].strip())
            break

    info = {'ul_x': ul_x, 'ul_y': ul_y, 'lr_x': lr_x, 'lr_y': lr_y, 'utm_zone': utm_zone}
    return info

# reading into geo lat lons,
def get_fire_coords(filename, label):
    with open(filename) as f:
        data = json.load(f)

    #lats = data[label]['lats']
    #lons = data[label]['lons']
    lats_lons = data[label]['lats_lons']
    lats = [x[0] for x in lats_lons]
    lons = [x[1] for x in lats_lons]
    return lats, lons

# converting to UTM projected coordinates
def transform_coords(lats, lons, info):
    proj_lats = []
    proj_lons = []
    for i in range(len(lats)):
        temp = utm.from_latlon(lats[i], lons[i], info['utm_zone'])
        if (min(info['ul_x'], info['lr_x']) < temp[0] < max(info['ul_x'], info['lr_x']) \
            and min(info['ul_y'], info['lr_y'])< temp[1] < max(info['ul_y'], info['lr_y'])):
            proj_lats.append(temp[0])
            proj_lons.append(temp[1])

    return proj_lats, proj_lons

def crop_fires(proj_lats, proj_lons, band_image_dict, label, crop_size, info):
    x_range = abs(info['lr_x']-info['ul_x'])
    y_range = abs(info['ul_y']-info['lr_y'])

    height = np.asarray(band_image_dict['b4']).shape[0]
    width = np.asarray(band_image_dict['b4']).shape[1]

    pad_size = crop_size/2.0

    existing_px = []
    existing_py = []

    for i in range(len(proj_lats)):
        x = proj_lats[i]
        y = proj_lons[i]
        # counterintuitive but lat (x) corresponds to height
        # lat corresponds to width
        p_x = (x - min(info['ul_x'], info['lr_x'])) * width/(x_range)
        p_y = (y - min(info['ul_y'], info['lr_y'])) * height/(y_range)
        # take out if statement out if sure no duplicates
        #if (i == 0 or (all(np.asarray(existing_px)-p_x) > crop_size) and all((np.asarray(existing_py)-p_y) > crop_size)):
        crop(band_image_dict, p_x, p_y, pad_size, label, i, 1)

def crop_non_fires(proj_lats, proj_lons, band_image_dict, label, crop_size, info):
    x_range = abs(info['lr_x']-info['ul_x'])
    y_range = abs(info['ul_y']-info['lr_y'])

    height = np.asarray(band_image_dict['b4']).shape[0]
    width = np.asarray(band_image_dict['b4']).shape[1]

    pad_size = crop_size/2.0

    # one nonfire for each fire image
    for i in range(len(proj_lats)):
        # sampling from middle half to ensure no black borders
        p_x = np.random.randint(int(width/4), int(3*width/4), 1)[0]
        p_y = np.random.randint(int(height/4), int(3*height/4), 1)[0]
        # take the if statement out if sure no duplicates
        #if (i == 0 or (all(np.asarray(existing_px)-p_x) > crop_size) and all((np.asarray(existing_py)-p_y) > crop_size)):
        crop(band_image_dict, p_x, p_y, pad_size, label, i, 0)


def crop(band_image_dict, p_x, p_y, pad_size, label, index, is_fire):
    p_y = int(p_y)
    p_x = int(p_x)
    pad_size = int(pad_size)
     # crop dimensions (left, upper, right, lower)
    height = np.asarray(band_image_dict['b4']).shape[0]
    p_y = height - p_y

    temp_b4 = np.asarray(band_image_dict['b4'].crop((p_x - pad_size , p_y - pad_size , p_x + pad_size , p_y + pad_size)))
    temp_b3 = np.asarray(band_image_dict['b3'].crop((p_x - pad_size , p_y - pad_size , p_x + pad_size , p_y + pad_size)))
    temp_b2 = np.asarray(band_image_dict['b2'].crop((p_x - pad_size , p_y - pad_size , p_x + pad_size , p_y + pad_size)))
    temp_b7 = np.asarray(band_image_dict['b7'].crop((p_x - pad_size , p_y - pad_size , p_x + pad_size , p_y + pad_size)))
    temp_b5 = np.asarray(band_image_dict['b5'].crop((p_x - pad_size , p_y - pad_size , p_x + pad_size , p_y + pad_size)))
    temp_b1 = np.asarray(band_image_dict['b1'].crop((p_x - pad_size , p_y - pad_size , p_x + pad_size , p_y + pad_size)))
    rgb = np.stack([temp_b4, temp_b3, temp_b2], -1)
    ir = np.stack([temp_b7, temp_b5, temp_b1], -1)

    outfile_name = label + '_' + str(index) + '_' + str(is_fire)

    outfile_name = os.path.join(OUTPUT_PATH, outfile_name)
    np.save(outfile_name + '_rgb', rgb)
    np.save(outfile_name +'_ir', ir)
    rgb_im = Image.fromarray(rgb)
    rgb_im.save(outfile_name + '_rgb.jpg')
    ir_im = Image.fromarray(ir)
    ir_im.save(outfile_name + '_ir.jpg')

def save_ir_image(band_image_dict, label):
    temp_b7 = np.asarray(band_image_dict['b7'])
    temp_b5 = np.asarray(band_image_dict['b5'])
    temp_b1 = np.asarray(band_image_dict['b1'])
    ir = np.stack([temp_b7, temp_b5, temp_b1], -1)
    ir_im = Image.fromarray(ir)
    ir_im.save(label + '_ir.jpg')
    return ir

def plot_fires(ir, band_image_dict, proj_lats, proj_lons, info):
    # latidue goes up down, index 0
    # longitude = across, index 0

    x_range = abs(info['lr_x']-info['ul_x'])
    y_range = abs(info['ul_y']-info['lr_y'])

    height = np.asarray(band_image_dict['b4']).shape[0]
    width = np.asarray(band_image_dict['b4']).shape[1]

    plt.imshow(ir)

    for i in range(len(proj_lats)):
        x = proj_lats[i]
        y = proj_lons[i]
        p_x = (x - min(info['ul_x'], info['lr_x'])) * width/(x_range)
        p_y = (y - min(info['ul_y'], info['lr_y'])) * height/(y_range)
        # when plotting coordinates, x first, then y 
        plt.scatter(p_x, height-p_y, color = 'r',s=5, alpha=0.5)
        print 'x: ' + str(p_x) + ' y: ' + str(height-p_y)
    #plt.scatter(0,0,s=10)
    plt.show()

if __name__ == '__main__':
    crop_size = 224
    bands = [4, 3, 2, 7, 5, 1]

    labels_file = os.path.join(DEFAULT_DOWNLOAD_PATH, "landsat_fire_2017_scene_lists.json")
    fires_file = os.path.join(DEFAULT_DOWNLOAD_PATH, "landsat_fire_2017_meta.json")

    with open(labels_file, 'r') as f:
        labels = json.load(f)

    download_path = DEFAULT_DOWNLOAD_PATH

    ### for local testing:
    #labels = ['LC08_L1TP_041026_20170929_20171013_01_T1']
    labels = ['LC08_L1TP_033026_20171124_20171206_01_T1']

    for label in labels:
        image_dir = os.path.join(download_path, label)

        # getting strings for reading in each band
        band_path_dict = {}
        for i in bands:
            band_path_dict['b' + str(i)] = os.path.join(image_dir, label + '_B' + str(i) + '.TIF')

        meta_file = os.path.join(image_dir, label + '_MTL.txt')

        # ensuring all bands and MTL files are present
        # takes longest time to load bands in
        if (all([os.path.exists(v) for v in band_path_dict.values()]) and os.path.exists(image_dir) and os.path.exists(meta_file)):
            band_image_dict = {}
            for i in bands:
                temp = cv2.imread(band_path_dict['b' + str(i)], 0)
                band_image_dict['b' + str(i)] = Image.fromarray(temp)
                print i

            print "Cropping image {}".format(label)

            info = get_metainfo(meta_file)
            print info
            lats, lons = get_fire_coords(fires_file, label)
            proj_lats, proj_lons = transform_coords(lats, lons, info)
 
            crop_fires(proj_lats, proj_lons, band_image_dict, label, crop_size, info)
            crop_non_fires(proj_lats, proj_lons, band_image_dict, label, crop_size, info)

            ### for visualiztion purposes, can ignore
            #ir = save_ir_image(band_image_dict,label)
            #plot_fires(ir, band_image_dict, proj_lats, proj_lons, info)

        else:             
            print "Image directory or file or all bands {} does not exist".format(image_dir)













