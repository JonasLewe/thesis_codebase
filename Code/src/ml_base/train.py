import os
from collections import Counter
from ml_base.models import define_base_model, define_vgg_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# run the test harness for evaluating a model
def train_model(img_root_dir, image_size, callbacks=[], additional_metrics=False, model_name='base', epochs=50, batch_size=32, index=22, verbose=1, learning_rate=0.001):
    # define model
    if model_name == 'base':
        model = define_base_model(additional_metrics=additional_metrics, learning_rate=learning_rate)
    elif model_name == 'vgg':
        model = define_vgg_model(additional_metrics=additional_metrics, learning_rate=learning_rate)
    
    # create data generators
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                       #rotation_range=40,
                                       #width_shift_range=0.2,
                                       #height_shift_range=0.2,
                                       #shear_range=0.2,
                                       #zoom_range=0.2,
                                       #brightness_range=[0.1,1],
                                       #horizontal_flip=True,
                                       #fill_mode='nearest'
                                      )
    
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # prepare iterators
    train_generator = train_datagen.flow_from_directory(os.path.join(img_root_dir, 'train'),
                                                 class_mode='binary',
                                                 color_mode='rgb',
                                                 batch_size=batch_size,
                                                 target_size=image_size)
    
    val_generator = val_datagen.flow_from_directory(os.path.join(img_root_dir, 'validation'),
                                               class_mode='binary', 
                                               color_mode='rgb',
                                               batch_size=batch_size, 
                                               target_size=image_size)
    
    # calculate class weights
    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}
    
    ### Debug info ###
    print(f"\nClasses: {train_generator.class_indices}")
    print(f"Class Weights: {class_weights}")
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