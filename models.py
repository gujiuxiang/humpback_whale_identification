from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras import applications
from keras.utils.data_utils import get_file
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Conv2D, MaxPooling2D, Reshape, Flatten, Input, merge, Lambda, Dropout

from keras.applications.imagenet_utils import _obtain_input_shape
from keras.layers.normalization import BatchNormalization
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import tensorflow as tf


def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([shape[:1] // parts, shape[1:]], axis=0)
        stride = tf.concat([shape[:1] // parts, shape[1:] * 0], axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx': i, 'parts': gpu_count})(x)
                    inputs.append(slice_n)

                outputs = model(inputs)

                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)

def build_model_128():
    print('Create the network ...')
    resize_shape = (128, 128, 3)
    convnet = Sequential(name='convnet')
    convnet.add(Conv2D(filters=8, input_shape=(
        resize_shape[0], resize_shape[1], resize_shape[2],), kernel_size=5, activation='relu', name='conv_1'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_1'))
    convnet.add(Conv2D(filters=12, kernel_size=3, activation='relu', name='conv_2'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_2'))
    convnet.add(Conv2D(filters=16, kernel_size=3, activation='relu', name='conv_3'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_3'))
    convnet.add(Conv2D(filters=20, kernel_size=3, activation='relu', name='conv_4'))
    convnet.add(Conv2D(filters=32, kernel_size=3, activation='relu', name='conv_5'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_4'))
    convnet.add(Flatten())
    convnet.add(Dense(units=128, activation='relu', name='dense_1'))
    #convnet.add(BatchNormalization())
    #convnet.add(Dropout(0.5))
    convnet.add(Dense(units=128, activation='relu', name='dense_2'))
    #convnet.add(BatchNormalization())
    #convnet.add(Dropout(0.5))
    convnet.add(Dense(units=64, activation='relu', name='dense_3'))
    return convnet

def build_model_256():
    print('Create the network ...')
    resize_shape = (256, 256, 3)
    convnet = Sequential(name='convnet')
    convnet.add(Conv2D(filters=8, input_shape=(
        resize_shape[0], resize_shape[1], resize_shape[2],), kernel_size=5, activation='relu', name='conv_1'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_1'))
    convnet.add(Conv2D(filters=12, kernel_size=3, activation='relu', name='conv_2'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_2'))
    convnet.add(Conv2D(filters=16, kernel_size=3, activation='relu', name='conv_3'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_3'))
    convnet.add(Conv2D(filters=20, kernel_size=3, activation='relu', name='conv_4'))
    convnet.add(Conv2D(filters=32, kernel_size=3, activation='relu', name='conv_5'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_4'))
    convnet.add(Flatten())
    convnet.add(Dense(units=256, activation='relu', name='dense_1'))
    #convnet.add(BatchNormalization())
    #convnet.add(Dropout(0.5))
    convnet.add(Dense(units=128, activation='relu', name='dense_2'))
    #convnet.add(BatchNormalization())
    #convnet.add(Dropout(0.5))
    convnet.add(Dense(units=64, activation='relu', name='dense_3'))
    return convnet

def build_model_vgg16():
    WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

    print('Create the network ...')
    resize_shape = (256, 256, 3)
    convnet = Sequential(name='convnet')
    # Block 1
    #convnet.add(Flatten(input_shape=(resize_shape[0], resize_shape[1], resize_shape[2],)))
    convnet.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(256, 256, 3)))
    convnet.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    convnet.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    convnet.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    convnet.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    convnet.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    convnet.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    convnet.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    convnet.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    convnet.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    convnet.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    convnet.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    convnet.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    convnet.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    convnet.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    convnet.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    convnet.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    convnet.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                            TF_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')
    convnet.load_weights(weights_path)

    # Classification block
    convnet.add(Flatten(name='flatten'))
    convnet.add(Dense(4096, activation='relu', name='fc1'))
    convnet.add(Dense(1024, activation='relu', name='fc2'))
    convnet.add(Dense(64, activation='relu', name='dense_3'))

    return convnet

def build_model_vgg16_edge():
    print('Create the network ...')
    resize_shape = (128, 128, 3)
    convnet = Sequential(name='convnet')
    convnet.add(Conv2D(filters=8, input_shape=(
        resize_shape[0], resize_shape[1], resize_shape[2],), kernel_size=5, activation='relu', name='conv_1'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_1'))
    convnet.add(Conv2D(filters=12, kernel_size=3, activation='relu', name='conv_2'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_2'))
    convnet.add(Conv2D(filters=16, kernel_size=3, activation='relu', name='conv_3'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_3'))
    convnet.add(Conv2D(filters=20, kernel_size=3, activation='relu', name='conv_4'))
    convnet.add(Conv2D(filters=32, kernel_size=3, activation='relu', name='conv_5'))
    convnet.add(MaxPooling2D(pool_size=2, name='pool_4'))
    convnet.add(Flatten())
    convnet.add(Dense(units=128, activation='relu', name='dense_1'))
    convnet.add(Dense(units=128, activation='relu', name='dense_2'))
    convnet.add(Dense(units=64, activation='relu', name='dense_3'))
    return convnet