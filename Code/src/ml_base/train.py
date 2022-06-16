import os
from collections import Counter
from utils.img import hist_eq
from utils import joined_generator
from ml_base import models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input


# run the test harness for evaluating a model
def train_model_single_view(img_root_dir,
                image_size,
                callbacks=[],
                verbose_metrics=False,
                model_name='base',
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
        model = models.define_base_model_single_xpl(learning_rate=learning_rate,
                                    verbose_metrics=verbose_metrics,
                                    image_size=image_size,
                                    dropout=dropout,
                                    regularization=regularization,
                                    num_hidden_layers=num_hidden_layers,
                                    )

    elif model_name == 'vgg':
        # model = define_vgg_model(verbose_metrics=verbose_metrics, image_size=image_size, learning_rate=learning_rate)
        model = models.define_vgg_model_functional(learning_rate=learning_rate, verbose_metrics=verbose_metrics, regularization=regularization)
    
    elif model_name == 'resnet50':
        model = models.define_resnet50_model(learning_rate=learning_rate, verbose_metrics=verbose_metrics, regularization=regularization)
    
    elif model_name == 'unet':
        #model = models.define_unet_model()
        model = models.unet()

    # create data generators
    train_datagen = ImageDataGenerator(# preprocessing_function=hist_eq,
                                        rescale=1.0/255.0,
                                        # preprocessing_function=preprocess_input, # testing for vgg, remove afterwards!
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
                                                        #shuffle=True,
                                                        )

    val_generator = val_datagen.flow_from_directory(os.path.join(img_root_dir, 'validation'),
                                                    class_mode='binary', 
                                                    color_mode='rgb',
                                                    batch_size=batch_size, 
                                                    target_size=image_size,
                                                    #shuffle=True,
                                                    )

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
                        #class_weight=class_weights,
                        epochs=epochs, 
                        verbose=verbose,
                        callbacks=callbacks)
    
    return model, history




# run the test harness for evaluating a model
def train_model_multi_view(img_root_dir,
                image_size,
                callbacks=[],
                verbose_metrics=False,
                model_name='base',
                early_fusion=False,
                fusion_technique=2,
                input_size=6,
                epochs=40,
                batch_size=32,
                index=22,
                verbose=1,
                learning_rate=0.01,
                dropout=0.2,
                regularization=False,
                num_hidden_layers=1):
    

    ppl_root_dir0 = f"{img_root_dir}_ppl_0"
    ppl_root_dir30 = f"{img_root_dir}_ppl_30"
    ppl_root_dir60 = f"{img_root_dir}_ppl_60"
    ppl_root_dir_merged = f"{img_root_dir}_ppl"

    xpl_root_dir0 = f"{img_root_dir}_xpl_0"
    xpl_root_dir30 = f"{img_root_dir}_xpl_30"
    xpl_root_dir60 = f"{img_root_dir}_xpl_60"
    xpl_root_dir_merged = img_root_dir

    train_datagen = ImageDataGenerator(# preprocessing_function=hist_eq,
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


    val_datagen = ImageDataGenerator(rescale=1.0/255.0) # use rescale to normalize pixel values in input data
    

    ppl_train_generator0 = train_datagen.flow_from_directory(os.path.join(ppl_root_dir0, 'train'),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )
    
    ppl_train_generator30 = train_datagen.flow_from_directory(os.path.join(ppl_root_dir30, 'train'),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )

    ppl_train_generator60 = train_datagen.flow_from_directory(os.path.join(ppl_root_dir60, 'train'),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )
    
    ppl_train_generator_merged = train_datagen.flow_from_directory(os.path.join(ppl_root_dir_merged, 'train'),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )

    ppl_val_generator0 = val_datagen.flow_from_directory(os.path.join(ppl_root_dir0, 'validation'),
                                                    class_mode='binary', 
                                                    color_mode='rgb',
                                                    batch_size=batch_size, 
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )

    ppl_val_generator30 = val_datagen.flow_from_directory(os.path.join(ppl_root_dir30, 'validation'),
                                                    class_mode='binary', 
                                                    color_mode='rgb',
                                                    batch_size=batch_size, 
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )

    ppl_val_generator60 = val_datagen.flow_from_directory(os.path.join(ppl_root_dir60, 'validation'),
                                                    class_mode='binary', 
                                                    color_mode='rgb',
                                                    batch_size=batch_size, 
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )

    ppl_val_generator_merged = val_datagen.flow_from_directory(os.path.join(ppl_root_dir_merged, 'validation'),
                                                class_mode='binary', 
                                                color_mode='rgb',
                                                batch_size=batch_size, 
                                                target_size=image_size,
                                                #shuffle=False
                                                )                                                

    xpl_train_generator0 = train_datagen.flow_from_directory(os.path.join(xpl_root_dir0, 'train'),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )

    xpl_train_generator30 = train_datagen.flow_from_directory(os.path.join(xpl_root_dir30, 'train'),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )

    xpl_train_generator60 = train_datagen.flow_from_directory(os.path.join(xpl_root_dir60, 'train'),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )

    xpl_train_generator_merged = train_datagen.flow_from_directory(os.path.join(xpl_root_dir_merged, 'train'),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )                                               


    xpl_val_generator0 = val_datagen.flow_from_directory(os.path.join(xpl_root_dir0, 'validation'),
                                                class_mode='binary', 
                                                color_mode='rgb',
                                                batch_size=batch_size, 
                                                target_size=image_size,
                                                #shuffle=False
                                                )

    xpl_val_generator30 = val_datagen.flow_from_directory(os.path.join(xpl_root_dir30, 'validation'),
                                                class_mode='binary', 
                                                color_mode='rgb',
                                                batch_size=batch_size, 
                                                target_size=image_size,
                                                #shuffle=False
                                                )

    xpl_val_generator60 = val_datagen.flow_from_directory(os.path.join(xpl_root_dir60, 'validation'),
                                                class_mode='binary', 
                                                color_mode='rgb',
                                                batch_size=batch_size, 
                                                target_size=image_size,
                                                #shuffle=False
                                                )

    xpl_val_generator_merged = val_datagen.flow_from_directory(os.path.join(xpl_root_dir_merged, 'validation'),
                                                class_mode='binary', 
                                                color_mode='rgb',
                                                batch_size=batch_size, 
                                                target_size=image_size,
                                                #shuffle=False
                                                )
    
    # define model
    if model_name == 'base':
        if input_size == 2:
            if early_fusion:
                model = models.define_base_model_early_fusion_2_view(learning_rate=learning_rate, 
                                                            verbose_metrics=verbose_metrics, 
                                                            image_size=image_size, 
                                                            fusion_technique=fusion_technique)
            else:
                model = models.define_base_model_mid_fusion_2_view(learning_rate=learning_rate, 
                                                            verbose_metrics=verbose_metrics, 
                                                            image_size=image_size, 
                                                            fusion_technique=fusion_technique)

            train_generator = joined_generator.JoinedGenerator2View(ppl_train_generator_merged, xpl_train_generator_merged)
                                                
            val_generator = joined_generator.JoinedGenerator2View(ppl_val_generator_merged, xpl_val_generator_merged)
        

        elif input_size == 3:
            if early_fusion:
                model = models.define_base_model_early_fusion_3_view(learning_rate=learning_rate, 
                                                            verbose_metrics=verbose_metrics, 
                                                            image_size=image_size, 
                                                            fusion_technique=fusion_technique)
            else:
                model = models.define_base_model_mid_fusion_3_view(learning_rate=learning_rate, 
                                                            verbose_metrics=verbose_metrics, 
                                                            image_size=image_size, 
                                                            fusion_technique=fusion_technique)

            train_generator = joined_generator.JoinedGenerator3View(xpl_train_generator0, 
                                                    xpl_train_generator30, 
                                                    xpl_train_generator60)
                                                
            val_generator = joined_generator.JoinedGenerator3View(xpl_val_generator0,
                                                    xpl_val_generator30,
                                                    xpl_val_generator60)

        elif input_size == 6:
            if early_fusion:
                model = models.define_base_model_early_fusion_6_view(learning_rate=learning_rate, 
                                                            verbose_metrics=verbose_metrics, 
                                                            image_size=image_size, 
                                                            fusion_technique=fusion_technique)
            else:
                model = models.define_base_model_mid_fusion_6_view(learning_rate=learning_rate, 
                                                            verbose_metrics=verbose_metrics, 
                                                            image_size=image_size, 
                                                            fusion_technique=fusion_technique)


            train_generator = joined_generator.JoinedGenerator6View(ppl_train_generator0, 
                                                    ppl_train_generator30, 
                                                    ppl_train_generator60, 
                                                    xpl_train_generator0, 
                                                    xpl_train_generator30, 
                                                    xpl_train_generator60)
                                                
            val_generator = joined_generator.JoinedGenerator6View(ppl_val_generator0,
                                                    ppl_val_generator30,
                                                    ppl_val_generator60, 
                                                    xpl_val_generator0,
                                                    xpl_val_generator30,
                                                    xpl_val_generator60)


    elif model_name == 'vgg':
        # not sure yet if I want to test this
        pass

    
    # calculate class weights
    counter = Counter(xpl_train_generator0.classes)
    class_indices = xpl_train_generator0.class_indices


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
                        #class_weight=class_weights,
                        epochs=epochs, 
                        verbose=verbose,
                        callbacks=callbacks)
    
    return model, history




# run the test harness for evaluating a model
def train_model_multi_view_cats_vs_dogs(img_root_dir,
                image_size,
                callbacks=[],
                verbose_metrics=False,
                model_name='base',
                early_fusion=False,
                fusion_technique=2,
                input_size=6,
                epochs=40,
                batch_size=32,
                index=22,
                verbose=1,
                learning_rate=0.01,
                dropout=0.2,
                regularization=False,
                num_hidden_layers=1):
    

    cats_vs_dogs_root = img_root_dir
    cats_vs_dogs_flipped = f"{img_root_dir}_flipped"

    train_datagen = ImageDataGenerator(# preprocessing_function=hist_eq,
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


    val_datagen = ImageDataGenerator(rescale=1.0/255.0) # use rescale to normalize pixel values in input data
    

    cats_vs_dogs_train_generator = train_datagen.flow_from_directory(os.path.join(cats_vs_dogs_root, 'train'),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )

    cats_vs_dogs_flipped_train_generator = train_datagen.flow_from_directory(os.path.join(cats_vs_dogs_flipped, 'train'),
                                                    class_mode='binary',
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )

    cats_vs_dogs_val_generator = val_datagen.flow_from_directory(os.path.join(cats_vs_dogs_root, 'validation'),
                                                    class_mode='binary', 
                                                    color_mode='rgb',
                                                    batch_size=batch_size, 
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )

    cats_vs_dogs_flipped_val_generator = val_datagen.flow_from_directory(os.path.join(cats_vs_dogs_flipped, 'validation'),
                                                    class_mode='binary', 
                                                    color_mode='rgb',
                                                    batch_size=batch_size, 
                                                    target_size=image_size,
                                                    #shuffle=False
                                                    )
    
    
    
    if early_fusion:
        model = models.define_base_model_early_fusion_2_view(learning_rate=learning_rate, 
                                                    verbose_metrics=verbose_metrics, 
                                                    image_size=image_size, 
                                                    fusion_technique=fusion_technique)
    else:
        model = models.define_base_model_mid_fusion_2_view(learning_rate=learning_rate, 
                                                    verbose_metrics=verbose_metrics, 
                                                    image_size=image_size, 
                                                    fusion_technique=fusion_technique)

    train_generator = joined_generator.JoinedGenerator2View(cats_vs_dogs_train_generator, cats_vs_dogs_flipped_train_generator)
                                        
    val_generator = joined_generator.JoinedGenerator2View(cats_vs_dogs_val_generator, cats_vs_dogs_flipped_val_generator)
        


    
    # calculate class weights
    counter = Counter(cats_vs_dogs_train_generator.classes)
    class_indices = cats_vs_dogs_train_generator.class_indices


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
                        #class_weight=class_weights,
                        epochs=epochs, 
                        verbose=verbose,
                        callbacks=callbacks)
    
    return model, history