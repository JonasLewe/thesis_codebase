import os
from collections import Counter
from utils.img import hist_eq
from utils.joined_generator import JoinedGenerator
from ml_base.models import define_base_model, define_base_model_early_fusion, define_vgg_model, define_vgg_model_simple, create_dummy_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# run the test harness for evaluating a model
def train_model(img_root_dir,
                image_size,
                callbacks=[],
                verbose_metrics=False,
                model_name='base',
                early_fusion=False,
                late_fusion=False,
                epochs=40,
                batch_size=32,
                index=22,
                verbose=1,
                learning_rate=0.01,
                dropout=0.2,
                regularization=False,
                num_hidden_layers=1):
    # define model
    if model_name == 'base':
        if early_fusion:
            model = define_base_model_early_fusion(learning_rate=learning_rate, verbose_metrics=verbose_metrics, image_size=image_size)
        else:
            model = define_base_model(learning_rate=learning_rate,
                                      verbose_metrics=verbose_metrics,
                                      image_size=image_size,
                                      dropout=dropout,
                                      regularization=regularization,
                                      num_hidden_layers=num_hidden_layers,
                                      )

    elif model_name == 'vgg':
        # model = define_vgg_model(verbose_metrics=verbose_metrics, image_size=image_size, learning_rate=learning_rate)
        model = create_dummy_model(learning_rate=learning_rate)

    if early_fusion:
        xpl_root_dir = img_root_dir
        ppl_root_dir = f"{img_root_dir}_ppl"

        ppl_train_datagen = ImageDataGenerator(# preprocessing_function=hist_eq,
                                               rescale=1.0/255.0,
                                               # rotation_range=40,
                                               # width_shift_range=0.2,
                                               # height_shift_range=0.2,
                                               # shear_range=0.2,
                                               # zoom_range=0.2,
                                               # brightness_range=[0.1,1],
                                               # horizontal_flip=True,
                                               # fill_mode='nearest'
                                              )


        xpl_train_datagen = ImageDataGenerator(# preprocessing_function=hist_eq,
                                               rescale=1.0/255.0,
                                               # rotation_range=40,
                                               # width_shift_range=0.2,
                                               # height_shift_range=0.2,
                                               # shear_range=0.2,
                                               # zoom_range=0.2,
                                               # brightness_range=[0.1,1],
                                               # horizontal_flip=True,
                                               # fill_mode='nearest'
                                              )


        ppl_val_datagen = ImageDataGenerator(rescale=1.0/255.0) # use rescale to normalize pixel values in input data

        xpl_val_datagen = ImageDataGenerator(rescale=1.0/255.0) # use rescale to normalize pixel values in input data
        

        ppl_train_generator = ppl_train_datagen.flow_from_directory(os.path.join(ppl_root_dir, 'train'),
                                                     class_mode='binary',
                                                     color_mode='rgb',
                                                     batch_size=batch_size,
                                                     target_size=image_size,
                                                     shuffle=False)
    
        ppl_val_generator = ppl_val_datagen.flow_from_directory(os.path.join(ppl_root_dir, 'validation'),
                                                   class_mode='binary', 
                                                   color_mode='rgb',
                                                   batch_size=batch_size, 
                                                   target_size=image_size,
                                                   shuffle=False)


        xpl_train_generator = xpl_train_datagen.flow_from_directory(os.path.join(xpl_root_dir, 'train'),
                                                     class_mode='binary',
                                                     color_mode='rgb',
                                                     batch_size=batch_size,
                                                     target_size=image_size,
                                                     shuffle=False)
    
        xpl_val_generator = xpl_val_datagen.flow_from_directory(os.path.join(xpl_root_dir, 'validation'),
                                                   class_mode='binary', 
                                                   color_mode='rgb',
                                                   batch_size=batch_size, 
                                                   target_size=image_size,
                                                   shuffle=False)

        train_generator = JoinedGenerator(ppl_train_generator, xpl_train_generator)
                                                   
        val_generator = JoinedGenerator(ppl_val_generator, xpl_val_generator)

        
        # calculate class weights
        counter = Counter(xpl_train_generator.classes)
        class_indices = xpl_train_generator.class_indices

    elif late_fusion:
        # TODO
        pass

    else:
    
        # create data generators
        train_datagen = ImageDataGenerator(# preprocessing_function=hist_eq,
                                           rescale=1.0/255.0,
                                           rotation_range=40,
                                           # width_shift_range=0.2,
                                           # height_shift_range=0.2,
                                           # shear_range=0.2,
                                           # zoom_range=0.2,
                                           # brightness_range=[0.1,1],
                                           horizontal_flip=True,
                                           # fill_mode='nearest'
                                          )
    
        val_datagen = ImageDataGenerator(rescale=1.0/255.0) # use rescale to normalize pixel values in input data
    
        # prepare iterators
        train_generator = train_datagen.flow_from_directory(os.path.join(img_root_dir, 'train'),
                                                     class_mode='binary',
                                                     color_mode='rgb',
                                                     batch_size=batch_size,
                                                     target_size=image_size,
                                                     shuffle=True)
    
        val_generator = val_datagen.flow_from_directory(os.path.join(img_root_dir, 'validation'),
                                                   class_mode='binary', 
                                                   color_mode='rgb',
                                                   batch_size=batch_size, 
                                                   target_size=image_size,
                                                   shuffle=True)
    
        # calculate class weights
        counter = Counter(train_generator.classes)
        class_indices = train_generator.class_indices


    max_val = float(max(counter.values()))
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
    class_weights_printable = {k: round(v, 2) for k, v in class_weights.items()}
    
    ### Debug info ###
    print(f"\nClasses: {class_indices}")
    print(f"Class Weights: {class_weights_printable}")
    print(f"Class Distribution: {dict(counter)}\n")
    # fit model
    history = model.fit(train_generator, 
                        steps_per_epoch=len(train_generator), 
                        validation_data=val_generator, 
                        validation_steps=len(val_generator),
                        class_weight=class_weights,
                        epochs=epochs, 
                        verbose=verbose,
                        callbacks=callbacks)
    
    return model, history