import matplotlib.pyplot as plt

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
