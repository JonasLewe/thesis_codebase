import os
import tempfile
from email.mime import base
from ml_base.metrics import METRICS
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers.experimental.preprocessing import Rescaling # not available in tensorflow=2.1
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten, Concatenate


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


def weighted_loss(weight):
    # TODO
    pass


def define_base_model_early_fusion(learning_rate, image_size=(224, 224), verbose_metrics=False, weighted_loss=False, weight=0.5):
    input_ppl = Input(shape=(image_size[0], image_size[1], 3), name="input_ppl")
    input_xpl = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl")

    merge = Concatenate()([input_ppl, input_xpl])

    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(merge)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='last_conv_layer')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)

    output_layer = Dense(1, activation="sigmoid")(x)

    opt = Adam(learning_rate)

    if verbose_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']

    # metrics = tf.keras.metrics.CategoricalAccuracy()
    
    model = Model(inputs=[input_ppl, input_xpl], outputs=output_layer)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)

    return model


# define basic cnn model
def define_base_model_legacy(learning_rate, image_size=(224, 224), verbose_metrics=False, weighted_loss=False, weight=0.5):
    input_shape=(image_size[0], image_size[1], 3)
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
    if weighted_loss:
        model.add(Dense(1))
        loss = weighted_loss(weight=weight)
    else:
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


# base model implemented using Functional API
def define_base_model(learning_rate, image_size=(224, 224), verbose_metrics=False, dropout=0.2, regularization=False, num_hidden_layers=1, weighted_loss=False, weight=0.5):
    input_layer = Input(shape=(image_size[0], image_size[1], 3), name="input_layer")

    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout)(x)

    for _ in range(num_hidden_layers):
        x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(dropout)(x)

    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='last_conv_layer')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(dropout)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
    x = Dropout(0.5)(x)

    output_layer = Dense(1, activation="sigmoid")(x)

    # compile model
    # opt = SGD(lr=0.001, momentum=0.9)
    opt = Adam(learning_rate)

    # add regularization to current model
    if regularization:   
        model = add_regularization(model)

    # determine metrics used
    if verbose_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=metrics)
    # print(model.summary())
    return model


def define_vgg_model(learning_rate, image_size=(224, 224), verbose_metrics=False):
    input_shape=(image_size[0], image_size[1], 3)
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    # for layer in base_model.layers[:11]:
    for layer in base_model.layers:
        layer.trainable = False
        # layer.trainable = True # (train all layers from scratch)

    model = Sequential()
    #model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output) # copy all layers from base_model
    # model.add(Rescaling(1./255,input_shape = input_shape))
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

def define_vgg_model_simple(learning_rate, image_size=(224, 224), verbose_metrics=False):
    input_shape=(image_size[0], image_size[1], 3)
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


def create_dummy_model(learning_rate=0.01, input_shape=(224,224,3), n_classes=1, fine_tune=0):
    """
    This is a Functional API version of the Sequential VGG model above
    Compiles a model integrated with VGG16 pretrained layers
    
    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """
    
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = VGG16(include_top=False,
                     weights='imagenet', 
                     input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='sigmoid')(top_model)
    
    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    optimizer = Adam(learning_rate=learning_rate)

    # Compiles the model for training.
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model
