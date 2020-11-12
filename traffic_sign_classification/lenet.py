import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Activation, Flatten, Dense, AveragePooling2D

def LeNet(width, height, depth):
  """Generates the LeNet network. Consists of 2 Convolutional layers and 3 Fully Connected.

  Args:
    width (int): First paramater. The networks input's width.
    height (int): Second paramater. The networks input's height.
    depth (int): Third parameter. The networks input's depth
  Returns: 
    model: If successful, returns the LeNet model compiled and prints the models summary. 

    Raises an Error if args aren't integers.
  """ 

  model = keras.Sequential()
  model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(width,height,depth)))
  model.add(AveragePooling2D())
  model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
  model.add(AveragePooling2D())
  model.add(Flatten())
  model.add(Dense(units=120, activation='relu'))
  model.add(Dense(units=84, activation='relu'))
  model.add(Dense(units=43, activation = 'softmax'))

  model.summary()
  model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam', metrics=["accuracy"])

  return model
