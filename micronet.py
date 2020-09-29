import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Activation, Flatten, Dense, AveragePooling2D

def MicronNet(width, height, depth):
	"""Generates the MicronNet network. Consists 4 Convolutional layers and 3 Fully Connected.

	Args:
		width (int): First paramater. The networks input's width.
		height (int): Second paramater. The networks input's height.
		depth (int): Third parameter. The networks input's depth
	Returns: 
		model: If successful, returns the MicronNet model compiled and prints the models summary. 

		Raises an Error if args aren't integers.
	"""	

	model = Sequential()
	model.add(Conv2D(filters=1, input_shape=(width,height,depth), kernel_size=(1,1),padding='same'))
	model.add(BatchNormalization(epsilon=1e-6))
	model.add(Activation("relu"))

	model.add(Conv2D(filters=29, kernel_size=(5,5)))
	model.add(BatchNormalization(epsilon=1e-6))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

	model.add(Conv2D(filters=59, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization(epsilon=1e-6))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2)))

	model.add(Conv2D(filters=74, kernel_size=(3,3), padding='same'))
	model.add(BatchNormalization(epsilon=1e-6))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(300))
	model.add(BatchNormalization(epsilon=1e-6))
	model.add(Activation("relu"))
	model.add(Dense(300))
	model.add(Activation("relu"))
	model.add(Dense(43))
	model.add(Activation("softmax"))

	model.summary()
	model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam',	metrics=["accuracy"])

	return model