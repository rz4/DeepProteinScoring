'''
MNIST_GPU.py
Updated: 11/28/17

This script is used to benchmark GPU systems on the MNIST dataset using the Keras
nueral network library.

The network defined in this benchmark gets to 99.25% test accuracy after 12
epochs (3 seconds per epoch on Tesla P100). The network utilizes convolutional
layers to preform multi-class classification between the different handwritten
character images.

NOTE: To benchmark multi-GPUs, available GPUs must also be defined in
CUDA_VISIBLE_DEVICES environment variable.

'''
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.layers.core import Lambda
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.losses import categorical_crossentropy
from keras.layers import merge, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

nb_gpus = 1
epochs = 12
batch_size = 128

################################################################################

def make_parallel(model, gpu_count):
    '''
    Method distributes equal-length training batches to defined available GPUs.

    '''
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)): outputs_all.append([])

    # Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:
                inputs = []
                # Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape,
                            arguments={'idx':i, 'parts':gpu_count})(x)
                    inputs.append(slice_n)
                outputs = model(inputs)

                if not isinstance(outputs, list): outputs = [outputs]

                # Save all the outputs for merging back together later
                for l in range(len(outputs)): outputs_all[l].append(outputs[l])

    # Merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))

        return Model(input=model.inputs, output=merged)

def define_model():
    '''
    Method defines CNN graph. Network is made up of 2 convolutional layers,
    one with 32 feature maps and kernel size of 3x3, followed by another with 64
    feature maps and kernel size of 3x3. Both layers use the RELU activation
    function.

    A max pooling with kernel size of 2x2 is applied to the convolutions with a
    dropout of 0.25 followed by a flattening of feature maps.

    The previous layer is then connected to a fully-connected layer of 128 nuerons
    with RELU activation and 0.5 dropout. The output layer consist of a
    fully-connected layer containing 10 output neurons (for each MNIST character
    class) with softmax activation.

    Returns Keras Sequential Model Object.

    '''
    # Instantiate model and add convolutional layers
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Add pooling layer and flatten feature maps
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    # Add fully connected layer and output layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

def load_mnist():
    '''
    Method loads MNIST dataset. The images are 28x28 pixels with one channel.

    Load train/test split of MNIST dataset from Keras library.

    The input data values are normailized from byte values [0, 255] to float values
    [0.0, 1.0].

    The output data values are converted from integer class identifier to one-hot
    vector of class.

    Returns processed input and output data values for both train and test sets.

    '''
    # Input image dimensions
    img_rows, img_cols = 28, 28
    chans = 1

    # Load data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape data according to image dimensions
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, chans)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, chans)

    # Normalize data from [0,255] to [0.0, 1.0]
    x_train = x_train.astype('float32')/ 255.0
    x_test = x_test.astype('float32')/ 255.0

    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    '''
    Main method loads and processes MNIST dataset for CNN training.

    CNN model is then defined and complied. CNN graph is distributed among define_model
    GPUs for syncronous training.

    Train CNN model over the defined epochs with the defined batch_size using
    training and test sets.

    '''
    # Load training and test images
    x_train, x_test, y_train, y_test = load_mnist()

    # Define CNN model across GPUs and compile CNN model
    model = define_model()
    if nb_gpus > 1: make_parallel(model, nb_gpus)
    model.compile(loss=categorical_crossentropy, optimizer=Adadelta(),
                  metrics=['accuracy'])

    # Train CNN model
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_test, y_test))
