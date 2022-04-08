from ml_base.metrics import METRICS
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten


# define basic cnn model
def define_base_model(input_shape=(224, 224, 3), learn_rate=0.001, additional_metrics=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='last_conv_layer'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    # opt = SGD(lr=0.001, momentum=0.9)
    opt = Adam(learning_rate=learn_rate)
    if additional_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    return model


def define_vgg_model(input_shape=(224, 224, 3), index=22, additional_metrics=False):
    base_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    # base_model.trainable = False # freeze layers from pretrained model

    model = Sequential()
    #model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output) # copy all layers from base_model
    model.add(Rescaling(1./255,input_shape = input_shape))
    for layer in base_model.layers[1:-1]: # loop through convolution layers
        model.add(layer)
    model.add(base_model.get_layer('block5_pool')) # add final maxpooling2D layer
    model.add(Flatten())
    model.add(Dense(256,activation=('relu'))) 
    model.add(Dropout(.2))
    model.add(Dense(128,activation=('relu')))
    model.add(Dropout(.2))
    
    # Final layer
    model.add(Dense(1,activation=('sigmoid'),name="activation_1"))
    
    opt = Adam(learning_rate=0.001)
    if additional_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    
    return model