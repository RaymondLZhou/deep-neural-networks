import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

import processData
import plotImage
import assess

path = ('../../data/cats_and_dogs_filtered')

train_dir, validation_dir, train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir = processData.loadData(path)

total_train, total_val = processData.dataSize(train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir)

batch_size = 80
epochs = 501
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_data_gen, val_data_gen = plotImage.createImageSet(train_dir, validation_dir, batch_size, IMG_HEIGHT, IMG_WIDTH)

model = Sequential([
    Conv2D(16, 3, padding='same', kernel_initializer='he_uniform', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu', kernel_initializer='he_uniform'),
    Dense(1)
])

model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model.summary()
plot_model(model, to_file='../../images/model.png', show_shapes=True, show_layer_names=True)

history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

assess.assess(history, epochs)
