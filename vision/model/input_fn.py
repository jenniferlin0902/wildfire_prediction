"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf
from utils import is_fire
import numpy as np

#RGB_FILE_EXT = ".jpg"
INFRARED_FILE_EXT = "_B7.jpg"

def load_image(filename, size):
    image_string = tf.read_file(filename)
    print filename
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image = tf.image.resize_images(image, [size, size])

    return image

def _parse_function(filename, label, params):
    """Obtain the image from the filename (for both training and validation).
    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    images = []
    if params.use_rgb:
        images.append(load_image(filename + "_rgb.jpg", params.image_size))
    if params.use_ir:
        images.append(load_image(filename + "_ir.jpg", params.image_size))

    # concat all channels
    concat_images = tf.concat(images, axis=2)
    return concat_images, label, filename

#This is currently not use
def train_preprocess(image, labels, filename, use_random_flip):
#     """Image preprocessing for training.
#
#     Apply the following operations:
#         - Horizontally flip the image with probability 1/2
#         - Apply random brightness and saturation
#     """
#     if use_random_flip:
#         image = tf.image.random_flip_left_right(image)
     assert image.get_shape().as_list()[2] == 6
     image_rgb, image_ir = tf.split(image, 2, axis=2)
     image_rgb = tf.image.random_brightness(image_rgb, max_delta=32.0 / 255.0)
     image_ir = tf.image.random_brightness(image_ir, max_delta=32.0 / 255.0)
     image_rgb = tf.image.random_saturation(image_rgb, lower=0.5, upper=1.5)
     image_ir = tf.image.random_saturation(image_ir, lower=0.5, upper=1.5)
#      # Make sure the image is still in [0, 1]
     image_rgb = tf.clip_by_value(image_rgb, 0.0, 1.0)
     image_ir = tf.clip_by_value(image_ir, 0.0, 1.9)
#     #print image
     image = tf.concat([image_rgb, image_ir], axis=2)
     return image, labels, filename


def input_fn(is_training, filenames, labels, params):
    """Input function for the SIGNS dataset.

    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l: _parse_function(f, l, params)
    train_fn = lambda i, l, f: train_preprocess(i,l,f, params.use_random_flip)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels, filename = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op, 'filenames':filename}
    return inputs
