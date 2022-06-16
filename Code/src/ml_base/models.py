import os
import tempfile
from email.mime import base
from ml_base.metrics import METRICS
import tensorflow as tf
from tensorflow.keras import Input, layers
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers.experimental.preprocessing import Rescaling # not available in tensorflow=2.1
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten, Concatenate, Maximum, Average, Add, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.utils import plot_model


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

def weighted_bincrossentropy(true, pred, weights={0: 1, 1: 3.02}):
    """
    Calculates weighted binary cross entropy. The weights are fixed.
        
    This can be useful for unbalanced catagories.

    Adjust the weights here depending on what is required.

    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
        will be penalize 10 times as much as false negatives.
    """
    weight_zero = 1   #weights[0]
    weight_one = 3.02 #weights[1]

    # calculate the binary cross entropy
    bin_crossentropy = tf.keras.backend.binary_crossentropy(true, pred)

    # apply the weights
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return tf.keras.backend.mean(weighted_bin_crossentropy)


# base model implemented using Functional API
def define_base_model_single_xpl(learning_rate, image_size=(224, 224), verbose_metrics=False, dropout=0.2, regularization=False, num_hidden_layers=1, weighted_loss=False, weight=0.5):
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

    # determine metrics used
    if verbose_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']

    model = Model(inputs=input_layer, outputs=output_layer)

    # add regularization to current model
    if regularization:   
        model = add_regularization(model)

    model.compile(optimizer=opt, loss=weighted_bincrossentropy, metrics=metrics)
    # print(model.summary())
    return model


# multi-view model with fusion at the top
def define_base_model_early_fusion_2_view(learning_rate, image_size=(224, 224), fusion_technique=2, verbose_metrics=False, weighted_loss=False, weight=0.5):
    # This model performs early fusion right at the top layer
    # Fusion Techniques:
    # 0: Max Fusion
    # 1: Average Fusion
    # 2: Concatenation

    input_ppl = Input(shape=(image_size[0], image_size[1], 3), name="input_ppl")
    input_xpl = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl")

    # merge ppl and xpl parts
    if fusion_technique == 0:
        model_merge = Maximum()([input_ppl, input_xpl])
    elif fusion_technique == 1:
        model_merge = Average()([input_ppl, input_xpl])
    elif fusion_technique == 2:
        model_merge = Add()([input_ppl, input_xpl])
    else:
        model_merge = Concatenate()([input_ppl, input_xpl])

    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_merge)
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

    model.compile(optimizer=opt, loss=weighted_bincrossentropy, metrics=metrics)

    return model


# multi-view model with fusion before last convolutional layer
def define_base_model_mid_fusion_2_view(learning_rate, image_size=(224, 224), fusion_technique=2, verbose_metrics=False, weighted_loss=False, weight=0.5):
    # This model fuses the two views right before the 
    # Fusion Techniques:
    # 0: Max Fusion
    # 1: Average Fusion
    # 2: Concatenation
    
    # define inputs
    input_ppl = Input(shape=(image_size[0], image_size[1], 3), name="input_ppl")
    input_xpl = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl")

    # define ppl part of the network until last conv layer
    model_ppl = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_ppl)
    model_ppl = MaxPooling2D((2, 2))(model_ppl)
    model_ppl = Dropout(0.2)(model_ppl)

    model_ppl = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_ppl)
    model_ppl = MaxPooling2D((2, 2))(model_ppl)
    model_ppl = Dropout(0.2)(model_ppl)

    # define xpl part of the network until last conv layer
    model_xpl = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_xpl)
    model_xpl = MaxPooling2D((2, 2))(model_xpl)
    model_xpl = Dropout(0.2)(model_xpl)

    model_xpl = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_xpl)
    model_xpl = MaxPooling2D((2, 2))(model_xpl)
    model_xpl = Dropout(0.2)(model_xpl)

    # merge ppl and xpl parts
    if fusion_technique == 0:
        model_merge = Maximum()([model_ppl, model_xpl])
    elif fusion_technique == 1:
        model_merge = Average()([model_ppl, model_xpl])
    elif fusion_technique == 2:
        model_merge = Add()([model_ppl, model_xpl])
    else:
        model_merge = Concatenate()([model_ppl, model_xpl])

    model_merge = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='last_conv_layer')(model_merge)
    model_merge = MaxPooling2D((2, 2))(model_merge)
    model_merge = Dropout(0.2)(model_merge)

    model_merge = Flatten()(model_merge)
    model_merge = Dense(128, activation='relu', kernel_initializer='he_uniform')(model_merge)
    model_merge = Dropout(0.5)(model_merge)

    output_layer = Dense(1, activation="sigmoid")(model_merge)

    opt = Adam(learning_rate)

    if verbose_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']

    # metrics = tf.keras.metrics.CategoricalAccuracy()
    
    # define final model
    model = Model(inputs=[input_ppl, input_xpl], outputs=output_layer)

    # plot model
    # plot_model(model, to_file="my_awesome_path")

    model.compile(optimizer=opt, loss=weighted_bincrossentropy, metrics=metrics)

    return model


# multi-view model with fusion at the top
def define_base_model_early_fusion_3_view(learning_rate, image_size=(224, 224), fusion_technique=2, verbose_metrics=False, weighted_loss=False, weight=0.5):
    # This model performs early fusion right at the top layer
    # It fuses 3 different views of xpl images
    # Fusion Techniques:
    # 0: Max Fusion
    # 1: Average Fusion
    # 2: Concatenation

    # define inputs

    input_xpl0 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl0")
    input_xpl30 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl30")
    input_xpl60 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl60")

    # merge ppl and xpl parts
    if fusion_technique == 0:
        model_merge = Maximum()([input_xpl0, input_xpl30, input_xpl60])
    elif fusion_technique == 1:
        model_merge = Average()([input_xpl0, input_xpl30, input_xpl60])
    elif fusion_technique == 2:
        model_merge = Add()([input_xpl0, input_xpl30, input_xpl60])
    else:
        model_merge = Concatenate()([input_xpl0, input_xpl30, input_xpl60])

    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_merge)
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
    
    model = Model(inputs=[input_xpl0, input_xpl30, input_xpl60], outputs=output_layer)

    model.compile(optimizer=opt, loss=weighted_bincrossentropy, metrics=metrics)

    return model


def define_base_model_mid_fusion_3_view(learning_rate, image_size=(224, 224), fusion_technique=2, verbose_metrics=False, weighted_loss=False, weight=0.5):
    # Fusion Techniques:
    # 0: Max Fusion
    # 1: Average Fusion
    # 2: Concatenation
    
    # define inputs
    input_xpl0 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl0")
    input_xpl30 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl30")
    input_xpl60 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl60")


    # define xpl 0° part of the network until last conv layer
    model_xpl0 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_xpl0)
    model_xpl0 = MaxPooling2D((2, 2))(model_xpl0)
    model_xpl0 = Dropout(0.2)(model_xpl0)

    model_xpl0 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_xpl0)
    model_xpl0 = MaxPooling2D((2, 2))(model_xpl0)
    model_xpl0 = Dropout(0.2)(model_xpl0)


    # define xpl 30° part of the network until last conv layer
    model_xpl30 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_xpl30)
    model_xpl30 = MaxPooling2D((2, 2))(model_xpl30)
    model_xpl30 = Dropout(0.2)(model_xpl30)

    model_xpl30 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_xpl30)
    model_xpl30 = MaxPooling2D((2, 2))(model_xpl30)
    model_xpl30 = Dropout(0.2)(model_xpl30)

    # define xpl 60° part of the network until last conv layer
    model_xpl60 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_xpl60)
    model_xpl60 = MaxPooling2D((2, 2))(model_xpl60)
    model_xpl60 = Dropout(0.2)(model_xpl60)

    model_xpl60 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_xpl60)
    model_xpl60 = MaxPooling2D((2, 2))(model_xpl60)
    model_xpl60 = Dropout(0.2)(model_xpl60)

    # merge ppl and xpl parts
    if fusion_technique == 0:
        model_merge = Maximum()([model_xpl0, model_xpl30, model_xpl60])
    elif fusion_technique == 1:
        model_merge = Average()([model_xpl0, model_xpl30, model_xpl60])
    elif fusion_technique == 2:
        model_merge = Add()([model_xpl0, model_xpl30, model_xpl60])
    else:
        model_merge = Concatenate()([model_xpl0, model_xpl30, model_xpl60])

    model_merge = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='last_conv_layer')(model_merge)
    model_merge = MaxPooling2D((2, 2))(model_merge)
    model_merge = Dropout(0.2)(model_merge)

    model_merge = Flatten()(model_merge)
    model_merge = Dense(128, activation='relu', kernel_initializer='he_uniform')(model_merge)
    model_merge = Dropout(0.5)(model_merge)

    output_layer = Dense(1, activation="sigmoid")(model_merge)

    opt = Adam(learning_rate)

    if verbose_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']

    # metrics = tf.keras.metrics.CategoricalAccuracy()
    
    # define final model
    model = Model(inputs=[input_xpl0, input_xpl30, input_xpl60], outputs=output_layer)

    # plot model
    # plot_model(model, to_file="my_awesome_path")

    model.compile(optimizer=opt, loss=weighted_bincrossentropy, metrics=metrics)

    return model


# multi-view model with fusion at the top
def define_base_model_early_fusion_6_view(learning_rate, image_size=(224, 224), fusion_technique=2, verbose_metrics=False, weighted_loss=False, weight=0.5):
    # This model performs early fusion right at the top layer
    # Fusion Techniques:
    # 0: Max Fusion
    # 1: Average Fusion
    # 2: Concatenation

    # define inputs
    input_ppl0 = Input(shape=(image_size[0], image_size[1], 3), name="input_ppl0")
    input_ppl30 = Input(shape=(image_size[0], image_size[1], 3), name="input_ppl30")
    input_ppl60 = Input(shape=(image_size[0], image_size[1], 3), name="input_ppl60")

    input_xpl0 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl0")
    input_xpl30 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl30")
    input_xpl60 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl60")

    # merge ppl and xpl parts
    if fusion_technique == 0:
        model_merge = Maximum()([input_ppl0, input_ppl30, input_ppl60, input_xpl0, input_xpl30, input_xpl60])
    elif fusion_technique == 1:
        model_merge = Average()([input_ppl0, input_ppl30, input_ppl60, input_xpl0, input_xpl30, input_xpl60])
    elif fusion_technique == 2:
        model_merge = Add()([input_ppl0, input_ppl30, input_ppl60, input_xpl0, input_xpl30, input_xpl60])
    else:
        model_merge = Concatenate()([input_ppl0, input_ppl30, input_ppl60, input_xpl0, input_xpl30, input_xpl60])

    x = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_merge)
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
    
    model = Model(inputs=[input_ppl0, input_ppl30, input_ppl60, input_xpl0, input_xpl30, input_xpl60], outputs=output_layer)

    model.compile(optimizer=opt, loss=weighted_bincrossentropy, metrics=metrics)

    return model


def define_base_model_mid_fusion_6_view(learning_rate, image_size=(224, 224), fusion_technique=2, verbose_metrics=False, weighted_loss=False, weight=0.5):
    # Fusion Techniques:
    # 0: Max Fusion
    # 1: Average Fusion
    # 2: Concatenation
    
    # define inputs
    input_ppl0 = Input(shape=(image_size[0], image_size[1], 3), name="input_ppl0")
    input_ppl30 = Input(shape=(image_size[0], image_size[1], 3), name="input_ppl30")
    input_ppl60 = Input(shape=(image_size[0], image_size[1], 3), name="input_ppl60")

    input_xpl0 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl0")
    input_xpl30 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl30")
    input_xpl60 = Input(shape=(image_size[0], image_size[1], 3), name="input_xpl60")


    # define ppl 0° part of the network until last conv layer
    model_ppl0 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_ppl0)
    model_ppl0 = MaxPooling2D((2, 2))(model_ppl0)
    model_ppl0 = Dropout(0.2)(model_ppl0)

    model_ppl0 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_ppl0)
    model_ppl0 = MaxPooling2D((2, 2))(model_ppl0)
    model_ppl0 = Dropout(0.2)(model_ppl0)


    # define ppl 30° part of the network until last conv layer
    model_ppl30 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_ppl30)
    model_ppl30 = MaxPooling2D((2, 2))(model_ppl30)
    model_ppl30 = Dropout(0.2)(model_ppl30)

    model_ppl30 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_ppl30)
    model_ppl30 = MaxPooling2D((2, 2))(model_ppl30)
    model_ppl30 = Dropout(0.2)(model_ppl30)


    # define ppl 60° part of the network until last conv layer
    model_ppl60 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_ppl60)
    model_ppl60 = MaxPooling2D((2, 2))(model_ppl60)
    model_ppl60 = Dropout(0.2)(model_ppl60)

    model_ppl60 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_ppl60)
    model_ppl60 = MaxPooling2D((2, 2))(model_ppl60)
    model_ppl60 = Dropout(0.2)(model_ppl60)


    # define xpl 0° part of the network until last conv layer
    model_xpl0 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_xpl0)
    model_xpl0 = MaxPooling2D((2, 2))(model_xpl0)
    model_xpl0 = Dropout(0.2)(model_xpl0)

    model_xpl0 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_xpl0)
    model_xpl0 = MaxPooling2D((2, 2))(model_xpl0)
    model_xpl0 = Dropout(0.2)(model_xpl0)


    # define xpl 30° part of the network until last conv layer
    model_xpl30 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_xpl30)
    model_xpl30 = MaxPooling2D((2, 2))(model_xpl30)
    model_xpl30 = Dropout(0.2)(model_xpl30)

    model_xpl30 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_xpl30)
    model_xpl30 = MaxPooling2D((2, 2))(model_xpl30)
    model_xpl30 = Dropout(0.2)(model_xpl30)

    # define xpl 60° part of the network until last conv layer
    model_xpl60 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(input_xpl60)
    model_xpl60 = MaxPooling2D((2, 2))(model_xpl60)
    model_xpl60 = Dropout(0.2)(model_xpl60)

    model_xpl60 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(model_xpl60)
    model_xpl60 = MaxPooling2D((2, 2))(model_xpl60)
    model_xpl60 = Dropout(0.2)(model_xpl60)

    # merge ppl and xpl parts
    if fusion_technique == 0:
        model_merge = Maximum()([model_ppl0, model_ppl30, model_ppl60, model_xpl0, model_xpl30, model_xpl60])
    elif fusion_technique == 1:
        model_merge = Average()([model_ppl0, model_ppl30, model_ppl60, model_xpl0, model_xpl30, model_xpl60])
    elif fusion_technique == 2:
        model_merge = Add()([model_ppl0, model_ppl30, model_ppl60, model_xpl0, model_xpl30, model_xpl60])
    else:
        model_merge = Concatenate()([model_ppl0, model_ppl30, model_ppl60, model_xpl0, model_xpl30, model_xpl60])

    model_merge = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', name='last_conv_layer')(model_merge)
    model_merge = MaxPooling2D((2, 2))(model_merge)
    model_merge = Dropout(0.2)(model_merge)

    model_merge = Flatten()(model_merge)
    model_merge = Dense(128, activation='relu', kernel_initializer='he_uniform')(model_merge)
    model_merge = Dropout(0.5)(model_merge)

    output_layer = Dense(1, activation="sigmoid")(model_merge)

    opt = Adam(learning_rate)

    if verbose_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']

    # metrics = tf.keras.metrics.CategoricalAccuracy()
    
    # define final model
    model = Model(inputs=[input_ppl0, input_ppl30, input_ppl60, input_xpl0, input_xpl30, input_xpl60], outputs=output_layer)

    # plot model
    # plot_model(model, to_file="my_awesome_path")

    model.compile(optimizer=opt, loss=weighted_bincrossentropy, metrics=metrics)

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


def define_vgg_model_functional(learning_rate=0.01, input_shape=(224,224,3), verbose_metrics=False, regularization=False, n_classes=1, fine_tune=0):
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
    # let fine_tune be <=18
    # train last block: fine_tune=4
    # train last 2 blocks: fine_tune=8
    # train last 3 blocks: fine_tune=12
    # train all except first: fine_tune=15
    # train all blocks: fine_tune=18
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

    # add regularization to current model
    if regularization:   
        model = add_regularization(model)

    # determine metrics used
    if verbose_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']

    # Compiles the model for training.
    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy',
                  metrics=metrics)
    
    return model


def define_resnet50_model(learning_rate=0.01, input_shape=(224,224,3), verbose_metrics=False, regularization=False):
    # last_conv_layer = "conv5_block3_3_conv"
    resnet_base = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

    for layer in resnet_base.layers:
        layer.trainable = True

    #input_layer = Input(shape=input_shape, name="input_layer")

    x = resnet_base.output
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(1, activation="sigmoid")(x)

    model = Model(resnet_base.input, output_layer)

    optimizer = Adam(learning_rate=learning_rate)

    # determine metrics used
    if verbose_metrics:
        metrics = METRICS
    else:
        metrics = ['accuracy']

    model.compile(optimizer=optimizer, 
                  loss='binary_crossentropy',
                  metrics=metrics)

    return model


# defining U-Net


def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


def define_unet_model(input_shape=(224,224,3)):
    inputs = layers.Input(shape=input_shape)

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)

    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)

    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)

    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)

    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)

    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)

    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)

    # outputs
    outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax", name="last_conv_layer")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")

    return unet_model


def unet(input_size=(224,224,1), pretrained_weights=None):
    
    inputs = layers.Input(input_size)
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = layers.concatenate([drop4,up6], axis = 3)
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = layers.concatenate([conv3,up7], axis = 3)
    conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = layers.concatenate([conv2,up8], axis = 3)
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = layers.concatenate([conv1,up9], axis = 3)
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = layers.Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    x = GlobalAveragePooling2D()(conv10)
    output_layer = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, output_layer)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model