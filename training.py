import pandas as pd
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input


#####  handle data classification

# from distutils.dir_util import copy_tree

# train_folder = 'dataset/train'
# test_folder = 'dataset/test'

# # Make a new train folder with fresh fruits
# toDirectory = 'working/train/freshfruits';

# fromDirectory = train_folder + '/freshapples';
# copy_tree(fromDirectory, toDirectory);
# fromDirectory = train_folder + '/freshbanana';
# copy_tree(fromDirectory, toDirectory);
# fromDirectory = train_folder + '/freshoranges';
# copy_tree(fromDirectory, toDirectory);

# # Make a new train folder with rotten fruits
# toDirectory = 'working/train/rottenfruits';

# fromDirectory = train_folder + '/rottenapples';
# copy_tree(fromDirectory, toDirectory);
# fromDirectory = train_folder + '/rottenbanana';
# copy_tree(fromDirectory, toDirectory);
# fromDirectory = train_folder + '/rottenoranges';
# copy_tree(fromDirectory, toDirectory);

# # Make a new validation folder with fresh fruits
# toDirectory = 'working/validation/freshfruits';
# fromDirectory = test_folder + '/freshapples';
# copy_tree(fromDirectory, toDirectory);
# fromDirectory = test_folder + '/freshbanana';
# copy_tree(fromDirectory, toDirectory);
# fromDirectory = test_folder + '/freshoranges';
# copy_tree(fromDirectory, toDirectory);

# # Make a new validation folder with rotten fruits
# toDirectory = 'working/validation/rottenfruits';

# fromDirectory = test_folder + '/rottenapples';
# copy_tree(fromDirectory, toDirectory);
# fromDirectory = test_folder + '/rottenbanana';
# copy_tree(fromDirectory, toDirectory);
# fromDirectory = test_folder + '/rottenoranges';
# copy_tree(fromDirectory, toDirectory);


############################################

#base vgg model
base_model = keras.applications.VGG16(
    weights='imagenet',
    input_shape=(224, 224, 3),
    include_top=False);

base_model.trainable = False

# Create inputs with correct shape
inputs =  keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)

# Add pooling layer
x = keras.layers.GlobalAveragePooling2D()(x)

# Add final dense layer
outputs = keras.layers.Dense(1,activation = 'sigmoid')(x)

# Combine inputs and outputs 
model = keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(from_logits=False), metrics=[keras.metrics.BinaryAccuracy()])

datagen_train = ImageDataGenerator(
    samplewise_center=True,  # set mean to 0
    rotation_range=10,  # rotate images 
    zoom_range=0.1,  # zoom image
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # shift images vertically 
    horizontal_flip=True,  
    vertical_flip=True, 
    rescale=1./255,                           
    preprocessing_function=keras.applications.vgg16.preprocess_input
)  

datagen_valid = ImageDataGenerator(samplewise_center=True)

# load and iterate training dataset
train_it = datagen_train.flow_from_directory(
    'working/train/',
    target_size=[224, 224],
    color_mode="rgb",
    class_mode="binary",
    batch_size=12,

)

# load and iterate validation dataset
valid_it = datagen_valid.flow_from_directory(
    'working/validation/',
    target_size= [224, 224],
    color_mode="rgb",
    class_mode="binary"
)

import math

steps_per_epoch = math.ceil(train_it.samples / train_it.batch_size)
validation_steps = math.ceil(valid_it.samples / valid_it.batch_size)

model.fit(train_it,
        validation_data = valid_it,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs = 1)

model.evaluate(valid_it, steps=math.ceil(valid_it.samples/valid_it.batch_size))

model.save('fruit_classification_model.h5')

def show_image(image_path):
    image = mpimg.imread(image_path)
    print(image.shape)
    plt.imshow(image)


def make_predictions(image_path):
    show_image(image_path)
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,224,224,3)
    image = preprocess_input(image)
    preds = model.predict(image)
    return preds


def fresh_or_rotten(image_path):
    preds = make_predictions(image_path)
    if preds <= 0.5:
        print("It's Fresh! eat ahead.")
    else:
        print("It's Rotten, I wont recommend!")