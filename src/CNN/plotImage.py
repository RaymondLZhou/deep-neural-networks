import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plotImages(images_arr, sets, crops):
    fig, axes = plt.subplots(sets, crops, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plotImageSet(train_data_gen, sets, crops):
    augmented_images = [train_data_gen[0][0][i] for i in range (sets) for j in range(crops)]
    plotImages(augmented_images, sets, crops)   

def createImageSet(train_dir, validation_dir, batch_size, IMG_HEIGHT, IMG_WIDTH):
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
    
    plotImageSet(train_data_gen, 4, 5)
    
    image_gen_val = ImageDataGenerator(rescale=1./255)

    val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                    directory=validation_dir,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode='binary')
    
    return train_data_gen, val_data_gen
