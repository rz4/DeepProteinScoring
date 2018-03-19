'''
'''
from keras.models import Model
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Flatten, Dense
from keras.layers import Activation, Concatenate, Multiply, Add, Average

def TorsionNet_v1(nb_class):
    '''
    '''
    # Input
    x = Input(shape=(19, 19, 23))

    # Conv Block 1
    l = Conv2D(64, (3, 3), strides = (1,1), padding='valid')(x)
    l = Activation('relu')(l)
    l = Dropout(0.25)(l)
    l = MaxPooling2D((2,2))(l)

    # Conv Block 2
    l = Conv2D(64, (3, 3), strides = (1,1), padding='valid')(l)
    l = Activation('relu')(l)
    l = Dropout(0.25)(l)
    l = MaxPooling2D((2,2))(l)

    # Fully Connected Layer
    l = Flatten()(l)
    l = Dense(1024, activation='relu')(l)
    l = Dropout(0.5)(l)

    # Output
    y = Dense(nb_class, activation='softmax')(l)

    # Model Defintions
    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def PairwiseNet_v1(nb_class):
    '''
    '''
    # Input
    x = Input(shape=(23, 23, 8))

    # Conv Block 1
    l = Conv2D(64, (3, 3), padding='valid')(x)
    l = Activation('relu')(l)
    l = Dropout(0.25)(l)
    l = MaxPooling2D((2,2))(l)

    # Conv Block 2
    l = Conv2D(64, (3, 3), padding='valid')(l)
    l = Activation('relu')(l)
    l = Dropout(0.25)(l)
    l = MaxPooling2D((2,2))(l)

    # Fully Connected Layer
    l = Flatten()(l)
    l = Dense(1024, activation='relu')(l)
    l = Dropout(0.5)(l)

    # Output
    y = Dense(nb_class, activation='softmax')(l)

    # Model Defintions
    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics

def PairwiseNet_v2(nb_class):
    '''
    '''
    # Input
    x = Input(shape=(23, 23, 8))

    # Conv Block 1
    l = Conv2D(64, (3, 3), padding='valid')(x)
    l = Activation('relu')(l)
    l = Dropout(0.25)(l)
    l = MaxPooling2D((2,2))(l)

    # Conv Block 2
    l = Conv2D(64, (3, 3), padding='valid')(l)
    l = Activation('relu')(l)
    l = Dropout(0.25)(l)
    l = MaxPooling2D((2,2))(l)

    # Fully Connected Layer
    l = Flatten()(l)
    l = Dense(1024, activation='relu')(l)
    l = Dropout(0.5)(l)

    # Output
    y = Dense(nb_class, activation='softmax')(l)

    # Model Defintions
    model = Model(inputs=x, outputs=y)
    loss = categorical_crossentropy
    optimizer = Adam(lr=0.0001,decay=0.1e-6)
    metrics = [categorical_accuracy,]

    return model, loss, optimizer, metrics
