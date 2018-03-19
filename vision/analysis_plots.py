from sklearn.metrics import confusion_matrix
import numpy as np
import os
#import sys
import shutil

FILENAME = 'mislabel_output.txt'
OUTPUT_FOLDER = 'analysis/error_images1'


'''def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')'''

def make_false_folders(false_negatives, false_positives):
	fn_dst_dir = os.path.join(OUTPUT_FOLDER, 'false_negatives')
	fp_dst_dir = os.path.join(OUTPUT_FOLDER, 'false_positives')

	image_type = '_ir.jpg'

	if not os.path.exists('analysis'):
		os.makedirs('analysis')
	if not os.path.exists(OUTPUT_FOLDER):
		os.makedirs(OUTPUT_FOLDER)
	if not os.path.exists(fn_dst_dir):
		os.makedirs(fn_dst_dir)
	if not os.path.exists(fp_dst_dir):
		os.makedirs(fp_dst_dir)

	for image in false_negatives:
		shutil.copy(image + image_type, fn_dst_dir)
	for image in false_positives:
		shutil.copy(image + image_type, fp_dst_dir)

if __name__ == '__main__':
	'''data_dir = 'data'
	train_data_dir = os.path.join(data_dir, "train_images")
	dev_data_dir = os.path.join(data_dir, "dev_images")
	test_data_dir = os.path.join(data_dir, "test_images")'''

	with open(FILENAME) as f:
		data = f.readlines()

	file_names = []
	actual_labels = []
	predicted_labels = []

	# for copying a new folder?
	false_positives = []
	false_negatives = []

	for line in data:
		if not line.isspace():
			temp = line.rstrip('\n')
			# need two conditions since "step" also recorded
			if (temp.find('true') != -1):
				file_name = temp.replace('true', '')
				actual_labels.append(int(file_name[-1]))
				predicted_labels.append(int(file_name[-1]))
				file_names.append(file_name)
			elif (temp.find('false') != -1):
				file_name = temp.replace('false', '')
				l = int(file_name[-1])
				actual_labels.append(l)
				predicted_labels.append(int(not l))
				file_names.append(file_name)
				# means actually no fire
				if (l == 0):
					false_positives.append(file_name)
				else:
					false_negatives.append(file_name)


	print actual_labels
	print predicted_labels
	#print file_names
	#print false_positives
	#print false_negatives

	make_false_folders(false_negatives, false_positives)

	print confusion_matrix(actual_labels, predicted_labels).ravel()



