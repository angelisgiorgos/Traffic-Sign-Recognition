import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Activation, Flatten, Dense, AveragePooling2D

def custom_model(width, height, depth):
  """Generates the custom model network. Consists of 5 Convolutional layers and 3 Fully Connected.

  Args:
    width (int): First paramater. The networks input's width.
    height (int): Second paramater. The networks input's height.
    depth (int): Third parameter. The networks input's depth
  Returns: 
    model: If successful, returns the custom model compiled and prints the models summary. 

    Raises an Error if args aren't integers.
  """ 

  model = keras.Sequential()
  inputS = (height, width, depth)
  chanDim = -1
  model.add(Conv2D(8, (5, 5), padding="same",input_shape=inputS))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  model.add(Conv2D(16, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  
  model.add(Conv2D(16, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Conv2D(32, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  
  model.add(Conv2D(32, (3, 3), padding="same"))
  model.add(Activation("relu"))
  model.add(BatchNormalization(axis=chanDim))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  
  model.add(Flatten())
  model.add(Dense(128))
  model.add(Activation("relu"))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  
  model.add(Dense(43))
  model.add(Activation("softmax"))

  model.summary()
  model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam',  metrics=["accuracy"])


  return model
