import os
import tempfile
from email.mime import base
from ml_base.metrics import METRICS
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten


def add_regularization(model, regularizer=tf.keras.regularizers.l2(l2=0.0001)):
    if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
        print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
        return model

    for layer in model.layers:
        for attr in ['kernel_regularizer']:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json() 
    
    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    model.save_weights(tmp_weights_path)

    # Load model from config
    model = tf.keras.models.model_from_json(model_json)

    # Reload the model weights
    model.load_weights(tmp_weights_path, by_name=True)

    return model


# define basic cnn model
def define_base_model(learning_rate, input_shape=(224, 224, 3), verbose_metrics=False):
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
    opt = Adam(learning_rate)

    # add regularization to current model
    # model = add_regularization(model)

    if verbose_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    # print(model.summary())
    return model


def define_vgg_model(learning_rate, input_shape=(224, 224, 3), verbose_metrics=False):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers[:11]:
        layer.trainable = False
        # layer.trainable = True # (train all layers from scratch)

    model = Sequential()
    #model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output) # copy all layers from base_model
    model.add(Rescaling(1./255,input_shape = input_shape))
    # model.add(tf.keras.Input(shape=input_shape))
    for layer in base_model.layers[:-1]: # loop through convolution layers
        model.add(layer)
    model.add(base_model.get_layer('block5_pool')) # add final maxpooling2D layer
    model.add(Flatten())
    model.add(Dense(256,activation=('relu'))) 
    model.add(Dropout(.2))
    model.add(Dense(128,activation=('relu')))
    model.add(Dropout(.2))
    
    # Final layer
    model.add(Dense(1,activation=('sigmoid'),name="activation_1"))
    
    # opt = Adam(learning_rate)
    opt = RMSprop(learning_rate)
    # opt = SGD(lr=learning_rate, momentum=0.9)

    # add regularization to current model
    model = add_regularization(model)

    # opt = Adam(learning_rate)
    if verbose_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)

    # print(model.summary())
    
    return model

def define_vgg_model01(learning_rate, input_shape=(224, 224, 3), verbose_metrics=False):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = True

    
    base_model.add(Flatten())
    base_model.add(Dense(256,activation=('relu'))) 
    base_model.add(Dropout(.2))
    base_model.add(Dense(128,activation=('relu')))
    base_model.add(Dropout(.2))

    base_model.add(Dense(1,activation=('sigmoid'),name="activation_1"))
    opt = SGD(lr=learning_rate, momentum=0.9)
    base_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])

    # print(base_model.summary())
    return base_model