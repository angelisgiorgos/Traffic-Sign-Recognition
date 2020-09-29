import matplotlib.pyplot as plt
import os
import csv
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
import random
import cv2
import seaborn as sns
from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage import transform
from sklearn.metrics import confusion_matrix
import tensorflow.keras.backend as K
from lenet import LeNet
from custom_model import custom_model
from micronet import MicronNet
tf.test.is_gpu_available()
tf.test.gpu_device_name()

def resize_cv(im,size):
	"""Function that resizes all dataset's images. Used on training-validation and test set.

	Args:
		im (src): The first parameter includes the image that we have previously imported.
		size (int): Second paramater. The images new size. Suggested values: (32,48,64)

	Returns: 
		Image: If successful returns the new resized image (src) with new dimensions (size x size x 3). 

		Raises an Error if OpenCv haven't imported previously, if arguments are empty values

	"""
	return cv2.resize(im, (size , size), interpolation = cv2.INTER_LINEAR)

def test_images(rootpath,size):
	"""Function that reads all testing set's images. Firstly creates two empty lists. After reads, with Pandas read_csv function from rootpath's csv file all DataFrame's instances.
	Iterating with pd.iterrows function through all dataframes rows. With the plt.image function reads images values. After that reads from dataframe the images roi's and then resizes the image
	with the resize_cv function. Finally appends the img and classid to the two list that we initialized in the first step.

	Args:
		rootpath (src): The first parameter, includes the dataset's directory.
		size (int): Second paramater. The images new size. Suggested values: (32,48,64).

	Returns: 
		lists: If successful, returns the two lists, one with all the test set images and one with the test set classes id. 

		Raises an Error if OpenCv, Pandas, Matplotlib haven't imported previously.

		Raises an Error if the testing set directory isn't correct.
	"""	
	test_images = []
	test_labels = []
	in_dir = '/content/GTSRB/Final_Test/Images/' #if you are working in a different directory must change also this.
	csv_file = pd.read_csv(os.path.join(rootpath), sep=';')
	for row in csv_file.iterrows() :
		img_path = os.path.join(in_dir, row[1].Filename)
		img = plt.imread(img_path)
		img = img[row[1]['Roi.X1']:row[1]['Roi.X2'],row[1]['Roi.Y1']:row[1]['Roi.Y2'],:]
		img = resize_cv(img,size)
		test_images.append(img)
		test_labels.append(row[1].ClassId)

	return test_images, test_labels


def visualizations(x_train,y_train):
	"""Function that creates a visualization of training dataset's images with their classes. Implemented with matplotlib. 

	Args:
		x_train (NumPy array): The first parameter, includes the training set images.
		y_train (NumPy array): Second paramater. The images labels in numbers mode. Values between (0,42).

	Returns: 
		plot: If successful, plots a new image, that includes random images from the training set with their labels. 

		Raises an Error if Matplotlib haven't imported previously, successfully.

	"""		

	W= 10
	L=10
	fig, axes = plt.subplots(L,W, figsize = (15,15))
	axes = axes.ravel()
	num_training = len(x_train)
	for i in np.arange(0,W*L):
		index=np.random.randint(0, num_training)
		axes[i].imshow(x_train[index])
		axes[i].set_title(y_train[index], fontsize=11)
		axes[i].axis("off")
	plt.subplots_adjust(bottom= 0.001,top=0.9,hspace=0.8)

	fig = plt.figure(figsize=(12,4))


def samples_visuals(y_train,y_val,y_test):
	"""Function that creates a histogram of dataset's samples splitted based on their class. In the x axis appended the dataset's labes (values 0~42) In the y-axis appended the number each label's number of samples. 

	Args:
		y_train (NumPy array): First paramater. The training set images labels in numbers mode. Values between (0,42).
		y_val (NumPy array): Second paramater. The validation set images labels in numbers mode. Values between (0,42).
		y_train (NumPy array): Third paramater. The testing set images labels in numbers mode. Values between (0,42).

	Returns: 
		plot: If successful, plots a new image, that includes a histgram with the training/validation/testing set samples splitted based on their class. 

		Raises an Error if Matplotlib haven't imported previously, successfully.

	"""	
	n, bins, patches = plt.hist(y_train, 43)
	plt.figure(figsize=(12,4))
	cls=['Training', 'Validation', 'Test']
	plt.title("Number of Train/Val/Test samples") 
	plt.xlabel("Labels")
	plt.ylabel("Samples")
	plt.hist([y_train,y_val,y_test], bins, stacked=True, label=cls)
	plt.show()

def evaluation_vis(History, string):
	"""Function generated in order to visualize the training/validation loss and accuracy diagrams.

	Args:
		History (TensorFlow class): First paramater. Contains model's training history. Includes four keys {loss, val_loss, accuracy, val_accuracy}
		string (str): Second paramater. Specific value, loss or accuracy. Specifies the metric's diagram.
	Returns: 
		plot: If successful, plots a metrics diagram, that includes training and validation loss or accuracy graphs. 

		Raises an Error if Matplotlib haven't imported previously, successfully and if the string variable is incorrect.

	"""
	N = np.arange(0, 40)
	plt.style.use("ggplot")
	plt.figure(figsize = (8,8))
	plt.plot(N, History.history[string], label="train "+string)
	plt.plot(N, History.history["val_"+string], label="val "+ string)
	plt.title("Training and Validation " + string)
	plt.xlabel("Epoch")
	plt.ylabel("Loss/Val" + string)
	plt.legend(loc="lower left")

def conf_matrix(y_true, pr_class):
	"""Generates confustion matrix to evaluate our model's performance.

	Args:
		y_true (NumPy array): First paramater. Contains The testing set images labels in numbers mode.
		pr_class (NumPy array): Second paramater. Includes predicted class value which is the result of Keras evaluate function.
	Returns: 
		plot: If successful, plots a confusion matrix. 

		Raises an Error if Matplotlib haven't imported previously.

	"""
	cm = confusion_matrix(y_true, pr_class)
	plt.figure(figsize = (15,15))
	sns.heatmap(cm, annot=True)

def img_rec(x_test, pr_class, y_true):
	"""Creates an image that includes testing set images with their correct and predicted class.

	Args:
		x_test (NumPy array): First paramater. Contains The training set images in converted in NumPy array.
		pr_class (NumPy array): Second paramater. Includes predicted class value which is the result of Keras evaluate function.
		y_true (NumPy array): Third parameter. Testing set classes values
	Returns: 
		plot: If successful, plots an image that includes random testing set images with their correct and predicted class. 

		Raises an Error if Matplotlib haven't imported previously.
	"""

	fig,axes = plt.subplots(5,5,figsize=(18,18))
	axes = axes.ravel()
	for i in np.arange(0,25):
		axes[i].imshow(x_test[i])
		axes[i].set_title("Prediction = {}\n True={}".format(pr_class[i], y_true[i]))
		axes[i].axis("off")

def eval_metrics(x_test,y_true):
	"""Evaluates models performance using metrics, accuracy, precision, recall, f1_score.

	Args:
		x_test (NumPy array): First paramater. Contains The training set images in converted in NumPy array.
		y_true (NumPy array): Second parameter. Testing set classes values
	Returns: 
		string: If successful, plots classification report for testing set in order to evaluate models perfomance. 

		Raises an Error if scikit learn classification report method haven't imported previously.
	"""

	yhat_probs = model.predict(x_test, verbose=0)
	yhat_classes = model.predict_classes(x_test, verbose=0)
	accuracy = accuracy_score(y_true, yhat_classes)
	print('Accuracy: %f' % accuracy)
	from sklearn.metrics import classification_report
	print(classification_report(y_true, yhat_classes))


data_dir = os.path.abspath('/content/GTSRB/Final_Training/Images')
os.path.exists(data_dir)

#Data preprocessing for first two models.

list_images = []
output = []
for dir in os.listdir(data_dir):
	if dir == '.DS_Store' :
		continue
		inner_dir = os.path.join(data_dir, dir)
		csv_file = pd.read_csv(os.path.join(inner_dir,"GT-" + dir + '.csv'), sep=';')
		for row in csv_file.iterrows() :
			img_path = os.path.join(inner_dir, row[1].Filename)
			img = plt.imread(img_path)
			img = img[row[1]['Roi.X1']:row[1]['Roi.X2'],row[1]['Roi.Y1']:row[1]['Roi.Y2'],:]
			img = resize_cv(img,size)
			list_images.append(img)
			output.append(row[1].ClassId)


x = np.asarray(list_images)
y = np.asarray(output)
x, y = shuffle(x, y)

split_size = int(x.shape[0]*0.75)
x_train, x_val= x[:split_size], x[split_size:]
y_train, y_val = y[:split_size], y[split_size:]

split_size = int(x.shape[0]*0.75)
x_train, x_val= x[:split_size], x[split_size:]
y_train, y_val = y[:split_size], y[split_size:]



'''
LeNet model training & evaluation
_________________________________________________
'''
print("Loading Data...")
x_train_gray = np.sum(x_train/3, axis=3, keepdims = True)
x_val_gray = np.sum(x_val/3, axis=3, keepdims = True)
x_test_gray = np.sum(x_test/3, axis=3, keepdims = True)
x_train_gray = (x_train_gray-128)/128
x_val_gray = (x_val_gray-128)/128
x_test_gray = (x_test_gray-128)/128

visualizations(x_train, y_train)

print("LeNet Training begins...")

model = LeNet(32,32,1)
History = model.fit(x_train_gray, 
                    y_train, 
                    batch_size=500, 
                    epochs=40, 
                    verbose =1, 
                    validation_data=(x_val_gray, y_val))


score = model.evaluate(x_test_gray, y_test, verbose=0)
print("Test Loss: {}".format(score[0]))
print("Test Accuracy: {}".format(score[1]))

tf.keras.utils.plot_model(model,to_file="lenet.png",)

evaluation_vis(History, 'loss')
evaluation_vis(History, 'accuracy')

pr_class = model.predict_classes(x_test_gray)

conf_matrix(y_test,pr_class)
img_rec(x_test,pr_class,y_test)
eval_metrics(x_test_gray, y_test)



'''
Custom model training & evaluation
_________________________________________________
'''

print('Custom model training begins...')

model2 = custom_model(32,32,3)
History2 = model2.fit(x_train, 
                    y_train, 
                    batch_size=500, 
                    epochs=40, 
                    verbose =1, 
                    validation_data=(x_val, y_val))

score2 = model2.evaluate(x_test, y_test, verbose =0)
print("Test Loss: {}".format(score[0]))
print("Test Accuracy: {}".format(score[1]))

tf.keras.utils.plot_model(model2,to_file="model2.png")

evaluation_vis(History2, 'loss')
evaluation_vis(History2, 'accuracy')

x_test2 = tf.cast(x_test, tf.float32)
yhat_class2 = model2.predict_classes(x_test2, verbose=0)

conf_matrix(y_test,yhat_class2)
img_rec(x_test,yhat_class2,y_test)
eval_metrics(x_test2, y_test)


'''
Custom model training & evaluation
_________________________________________________
'''
#Data preprocessing for third model.
print("Data Loading for third model...")
  
list_images2 = []
output2 = []
for dir in os.listdir(data_dir):
    if dir == '.DS_Store' :
        continue
    
    inner_dir = os.path.join(data_dir, dir)
    csv_file = pd.read_csv(os.path.join(inner_dir,"GT-" + dir + '.csv'), sep=';')
    for row in csv_file.iterrows() :
        img_path2 = os.path.join(inner_dir, row[1].Filename)
        img2 = plt.imread(img_path2)
        img2 = img2[row[1]['Roi.X1']:row[1]['Roi.X2'],row[1]['Roi.Y1']:row[1]['Roi.Y2'],:]
        img2 = resize_cv(img2,48)
        list_images2.append(img2)
        output2.append(row[1].ClassId)


x2 = np.asarray(list_images2)
y2 = np.asarray(output2)
x2, y2 = shuffle(x2, y2)


x_t2, y_t2 = test_images('/content/GTSRB/Final_Test/Images/GT-final_test.csv')
x_test2 = np.asarray(x_t2)
y_test2 = np.asarray(y_t2)
x2.shape
split_size2 = int(x2.shape[0]*0.75)
x_train2, x_val2= x2[:split_size2], x2[split_size2:]
y_train2, y_val2 = y2[:split_size2], y2[split_size2:]

print("Training set size:", len(x_train2))
print("Validation set size:", len(x_val2))
print("Test set size:", len(x_test2))

visualizations(x_train2, y_train2)

print('MicronNet training begins...')

model3 = MicronNet(48,48,3)
History3 = model3.fit(x_train2, 
                    y_train2, 
                    batch_size=500, 
                    epochs=50, 
                    verbose =1, 
                    validation_data=(x_val2, y_val2))

score2 = model3.evaluate(x_t2, y_t2, verbose =0)
print("Test Loss: {}".format(score2[0]))
print("Test Accuracy: {}".format(score2[1]))
tf.keras.utils.plot_model(model3,to_file="model3.png",)


evaluation_vis(History3, 'loss')
evaluation_vis(History3, 'accuracy')

x_test3 = tf.cast(x_t2, tf.float32)
yhat_class2 = model2.predict_classes(x_test3, verbose=0)

conf_matrix(y_t2,yhat_class2)
img_rec(x_test3,yhat_class2,y_t2)
eval_metrics(x_test3, y_t2)

