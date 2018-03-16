"""Define the model."""

import tensorflow as tf
from vgg16 import Vgg16

VGG_CONFIG = [
    [
        {"type":"Conv", "size":3, "filters":64},
        {"type": "Conv", "size": 3, "filters": 64},
        {"type": "Max", "size": 3},
    ],
    [
        {"type":"Conv", "size":3, "filters":128},
        {"type": "Max", "size": 2},
    ],
    [
        {"type": "Conv", "size": 3, "filters": 128},
        {"type": "Max", "size": 2},
    ],
    [
        {"type": "Flat"},
        {"type": "Fc", "size": 2048}
    ],
]

# TODO add bn layer to VGG net
def build_VGG(is_training, inputs, params):
    network_configs = VGG_CONFIG
    out = inputs
    for i, block in enumerate(network_configs):
        for j, network in enumerate(block):
            with tf.variable_scope("block_{}".format(i+1)):
                if network["type"] == "Conv":
                    # use ReLU and xavier_initializer by default
                    out = tf.contrib.layers.conv2d(out, network["filters"], network["size"], padding='same', scope="network_{}".format(j))
                elif network["type"] == "Max":
                    out = tf.layers.max_pooling2d(out, network["size"], network["size"])
                elif network["type"] == "Fc":
                    # use relu by default
                    out = tf.contrib.layers.fully_connected(out, network["size"],scope="network_{}".format(j))
                elif network["type"] == "Flat":
                    out = tf.contrib.layers.flatten(out)
                else:
                    raise "Invalid NN type arguement"
    return out

def build_simple(is_training, inputs, params):
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
    out = inputs
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, params.kernel_size(), padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

    out = tf.contrib.layers.flatten(out, scope="flatten_1")
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, num_channels * 8)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
    return out

def calculate_total_trainable():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters

def build_preetrained_VGG(inputs, params):
    vgg = Vgg16(trainable=params.trainable)
    out = vgg.build(inputs)

    # add another fc before output
    out = tf.contrib.layers.flatten(out, scope="flatten_1")
    out = tf.contrib.layers.fully_connected(out, 256)
    out = tf.layers.batch_normalization(out)

    # check trainable variables
    print "INFO : total trainable param = {}".format(calculate_total_trainable())
    return out

# added for case where both ir and rgb
def build_preetrained_VGG_double(inputs, params):
    inputs1, inputs2 = tf.split(inputs, [3, 3], 3)

    vgg1 = Vgg16(trainable=params.trainable)
    out1 = vgg1.build(inputs1)
    out1 = tf.contrib.layers.flatten(out1, scope="flatten_1")
    out1 = tf.contrib.layers.fully_connected(out1, 512)
    out1 = tf.layers.batch_normalization(out1)

    vgg2 = Vgg16(trainable=params.trainable)
    out2 = vgg2.build(inputs2)
    out2 = tf.contrib.layers.flatten(out2, scope="flatten_2")
    out2 = tf.contrib.layers.fully_connected(out2, 512)
    out2 = tf.layers.batch_normalization(out2)

    # combining together
    out1 = tf.contrib.layers.flatten(out1, scope="flatten_3")
    out2 = tf.contrib.layers.flatten(out2, scope="flatten_4")
    out = tf.concat([out1, out2], axis=1)

    out = tf.contrib.layers.fully_connected(out, 1024)
    out = tf.layers.batch_normalization(out)

    # check trainable variables
    print "INFO : total trainable param = {}".format(calculate_total_trainable())
    return out


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']
    print images.get_shape().as_list()
    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, params.num_channels]

    if params.model == "vgg_simple":
        out = build_VGG(is_training, images, params)
    elif (params.model == "vgg_pretrain" and params.num_channels == 3):
        out = build_preetrained_VGG(images, params)
    elif (params.model == "vgg_pretrain" and params.num_channels == 6):
        out = build_preetrained_VGG_double(images, params)
    else:
        out = build_simple(is_training, images, params)

    # always add an output layer
    with tf.variable_scope('output'):
        logits = tf.layers.dense(out, params.num_labels)

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
