import inspect
import os
import utils
import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16:
    def __init__(self, trainable=True, vgg16_npy_path=None, scope="Vgg16"):
        if vgg16_npy_path == None:
            vgg16_npy_path = os.path.join(os.getcwd(), "vgg16.npy")
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        self.trainable = trainable
        print("npy file loaded")
        self.var_dict = {}
        self.scope = scope

    def build(self, rgb):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr,3,64, "conv1_1", trainable=False)
        self.conv1_2 = self.conv_layer(self.conv1_1, 64,64, "conv1_2", trainable=False)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128,"conv2_1", trainable=False)
        self.conv2_2 = self.conv_layer(self.conv2_1,128, 128, "conv2_2", trainable=False)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128,256, "conv3_1", trainable=False)
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2", trainable=False)
        self.conv3_3 = self.conv_layer(self.conv3_2, 256,256,"conv3_3", trainable=False)
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3,256,512, "conv4_1", trainable=self.trainable)
        self.conv4_2 = self.conv_layer(self.conv4_1,512,512, "conv4_2",trainable=self.trainable)
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512,"conv4_3",trainable=self.trainable)
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4,512,512, "conv5_1", trainable=self.trainable)
        self.conv5_2 = self.conv_layer(self.conv5_1, 512,512,"conv5_2",trainable=self.trainable)
        self.conv5_3 = self.conv_layer(self.conv5_2, 512,512,"conv5_3", trainable=self.trainable)
        self.pool5 = self.max_pool(self.conv5_3,  'pool5')
        # 25088 = ((224 // (2 ** 5)) ** 2) * 512

        #self.fc6 = self.fc_layer(self.pool5,25088,4096, "fc6")
        #assert self.fc6.get_shape().as_list()[1:] == [4096]
        #self.relu6 = tf.nn.relu(self.fc6)

        #self.fc7 = self.fc_layer(self.relu6, "fc7")

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))
        # use relu 6 s output
        return self.pool5

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name, trainable=False):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name, trainable=trainable)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name, trainable=False):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters", trainable=trainable)

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable=trainable)
        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights", trainable=True)

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", trainable=True)

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name, trainable):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            print "WARNING: value {} not found in dict".format(name)
            value = initial_value

        if trainable:
            print "INFO: use trainable var for {} ".format(var_name)
            init = tf.constant_initializer(value)
            print value.shape
            with tf.variable_scope(self.scope):
                var = tf.get_variable(var_name,initializer=init, trainable=True, shape=value.shape)
        else:
            print "INFO: use constant for {}".format(var_name)
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count


# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1


if __name__ == "__main__":

    img3 = utils.load_image("../landsat_download_2016/LC80120292016131LGN01/LC80120292016131LGN00_B2.TIF")


    batch3 = img3.reshape((1, 224, 224, 3))

    with tf.device('/cpu:0'):
        with tf.Session() as sess:
            images = tf.placeholder("float", [None, 224, 224, 3])
            feed_dict = {images: batch3}

            vgg = Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)

            prob = sess.run(vgg.prob, feed_dict=feed_dict)
            print(prob)
            print_prob(prob[0], './synset.txt')
            print_prob(prob[1], './synset.txt')
            print_prob(prob[2], './synset.txt')
