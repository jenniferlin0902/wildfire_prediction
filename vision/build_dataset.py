"""Split the SIGNS dataset into train/dev/test and resize images to 64x64.

The SIGNS dataset comes in the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and dev sets.
Because we don't have a lot of images and we want that the statistics on the dev set be as
representative as possible, we'll take 20% of "train_signs" as dev set.
"""

import argparse
import random
import os
import numpy as np

from PIL import Image

from model.utils import is_fire
import shutil

SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/Volumes/DRIVE/landsat_cropped/', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    # train_data_dir = os.path.join(args.data_dir, 'train_images')
    # test_data_dir = os.path.join(args.data_dir, 'test_images')


    # set up dev/train/test dir
    data_dir = args.data_dir
    dev_dir = os.path.join(args.output_dir, "dev_images")
    train_dir = os.path.join(args.output_dir, "train_images")
    test_dir = os.path.join(args.output_dir, "test_images")
    print train_dir

    # Get the filenames in each directory (train and test)
    filenames = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    no_fire_files = []
    fire_files = []

    for f in filenames:
        print f
        if is_fire(os.path.basename(f)):
            fire_files.append(f)
        else:
            no_fire_files.append(f)
    print "found {} fire, {} no fire".format(len(fire_files), len(no_fire_files))
    src_filenames = fire_files + random.sample(no_fire_files, len(fire_files)*3)

    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    src_filenames.sort()
    random.shuffle(src_filenames)
    print "Splitting {} src files".format(len(src_filenames))

    split = [int(0.80*len(src_filenames)), int(0.90*len(src_filenames))] # train, dev, test
    print split
    print data_dir
    print train_dir
    for f in src_filenames[:split[0]]:
        print f
        shutil.move(os.path.join(data_dir, f), os.path.join(train_dir, f))
    for f in src_filenames[split[0]:split[1]]:
        shutil.move(os.path.join(data_dir, f), os.path.join(dev_dir, f))
    for f in src_filenames[split[1]:]:
        shutil.move(os.path.join(data_dir, f), os.path.join(test_dir, f))

    '''
    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_signs'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)
    '''
    print("Done building dataset")
