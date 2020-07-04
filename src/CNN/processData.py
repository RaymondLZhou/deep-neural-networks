import os

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
