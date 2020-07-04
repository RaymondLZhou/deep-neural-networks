import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

def loadData(path):
    train_dir = os.path.join(path, 'train')
    validation_dir = os.path.join(path, 'validation')

    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')

    return train_dir, validation_dir, train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir

def dataSize(train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir):
    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))

    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val

    print('Total training cat images:', num_cats_tr)
    print('Total training dog images:', num_dogs_tr)

    print('Total validation cat images:', num_cats_val)
    print('Total validation dog images:', num_dogs_val)

    print('Total training images:', total_train)
    print('Total validation images:', total_val)

    return total_train, total_val

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plotImageSet(train_data_gen):
    augmented_images = [train_data_gen[0][0][0] for i in range(5)]
    plotImages(augmented_images)  

    augmented_images = [train_data_gen[0][0][1] for i in range(5)]
    plotImages(augmented_images)  

def createImageSet(train_dir, validation_dir):
    image_gen_train = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=45,
                        width_shift_range=.15,
                        height_shift_range=.15,
                        horizontal_flip=True,
                        zoom_range=0.5
                        )
    
    train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                        directory=train_dir,
                                                        shuffle=True,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        class_mode='binary')
    
    plotImageSet(train_data_gen)
    
    image_gen_val = ImageDataGenerator(rescale=1./255)

    val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                    directory=validation_dir,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode='binary')
    
    return train_data_gen, val_data_gen

def assess(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

path = ('../../data/cats_and_dogs_filtered')

train_dir, validation_dir, train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir = loadData(path)

total_train, total_val = dataSize(train_cats_dir, train_dogs_dir, validation_cats_dir, validation_dogs_dir)

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_data_gen, val_data_gen = createImageSet(train_dir, validation_dir)

model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model_new.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

model_new.summary()

history = model_new.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

assess(history)
