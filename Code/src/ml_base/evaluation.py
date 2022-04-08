import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# plot diagnostic learning curves
def summarize_diagnostics(history, log_dir) -> None:
    # plot loss
    plt.subplot(211)
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue')
    plt.plot(history.history['val_loss'], color='orange')
    plt.legend(['train', 'val'])
    
    # plot accuracy
    plt.subplot(212)
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue')
    plt.plot(history.history['val_accuracy'], color='orange')
    plt.legend(['train', 'val']) 
    
    plt.subplots_adjust(top=1.6, right=1.2)
    
    if log_dir:
        plt.savefig(os.path.join(log_dir, "summarized_diagnostics.png"), bbox_inches='tight')
    else:
        plt.show()


def show_confusion_matrix(model, test_generator, log_dir) -> None:    
    test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)

    predictions = model.predict(test_generator, steps=test_steps_per_epoch)
    predicted_classes = [1 * (x[0]>=0.5) for x in predictions]
    
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys()) 
    
    cm = confusion_matrix(true_classes, predicted_classes)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    
    disp.plot(cmap=plt.cm.Blues)
    if log_dir:
        plt.savefig(os.path.join(log_dir, "confusion_matrix.png"))
    else:
        plt.show()
    
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report) 


def evaluate_model(img_root_dir, image_size, model, history, verbose=True, log_dir=""):
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(os.path.join(img_root_dir, 'test'),
                                               class_mode='binary',
                                               color_mode='rgb',
                                               target_size=image_size,
                                               batch_size=1,
                                               shuffle=False,)
    # calculate test set accuracy
    _, acc = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
    print('\nAccuracy on test set: %.3f' % (acc * 100.0))
    
    if verbose:
        # plot learning curves
        summarize_diagnostics(history, log_dir)

        # plot confusion matrix
        show_confusion_matrix(model, test_generator, log_dir)
    
    return acc


