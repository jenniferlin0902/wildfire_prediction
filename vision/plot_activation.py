import json
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.misc
import os

activation_file = "save_activation0.json"
save_path = "activation"

def plotNNFilter(units, layer_num, img):
    if units.ndim != 4:
        return
    print "plotting"
    filters = units.shape[3]
    #plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        #plt.subplot(n_rows, n_columns, i+1)
        #plt.figure()


        img_name = os.path.join(save_path, 'Layer{}Filter{}.jpg'.format(layer_num, i))
        normalized = units[0,:,:,i] / np.linalg.norm(units[0,:,:,i])
        #plt.imshow(normalized, cmap="gray")
        scipy.misc.imsave(img_name, normalized)


with open(activation_file, 'rb') as f:
    activations = json.load(f)
    for layer in activations:
        print layer
        print "layer {}, has dim {}".format(layer, np.array(activations[layer]).shape)
        plotNNFilter(np.array(activations[layer]), layer)