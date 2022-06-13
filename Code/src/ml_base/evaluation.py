import os
import numpy as np
from utils import joined_generator
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, precision_recall_fscore_support 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# plot diagnostic learning curves
def summarize_diagnostics_base(history, log_dir) -> None:
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
    
    plt.cla()


def summarize_diagnostics_verbose(history, log_dir) -> None:
    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(2, 3, 1)
    ax.plot(history.history["loss"], color="blue")
    ax.plot(history.history["val_loss"], color="orange")
    ax.set_title("Cross Entropy Loss")

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(history.history["accuracy"], color="blue")
    ax.plot(history.history["val_accuracy"], color="orange")
    ax.set_title("Classification Accuracy")

    ax = fig.add_subplot(2, 3, 3)
    ax.plot(history.history["precision"], color="blue")
    ax.plot(history.history["val_precision"], color="orange")
    ax.set_title("Classification Precision")

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(history.history["recall"], color="blue")
    ax.plot(history.history["val_recall"], color="orange")
    ax.set_title("Classification Recall")

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(history.history["auc"], color="blue")
    ax.plot(history.history["val_auc"], color="orange")
    ax.set_title("Classification AUC")


    fig.tight_layout()
    
    if log_dir:
        plt.savefig(os.path.join(log_dir, "summarized_diagnostics.png"), bbox_inches='tight')
    else:
        plt.show()
    
    plt.cla()


def show_results(model, test_generator, log_dir) -> None:    
    test_steps_per_epoch = np.math.ceil(test_generator.samples / test_generator.batch_size)

    predictions = model.predict(test_generator, steps=test_steps_per_epoch)
    #print(f"Predictions from evaluation.py:show_results: {predictions}")
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
    
    plt.cla()
    
    report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    print(report) 

    precision, recall, fscore, _ = precision_recall_fscore_support(true_classes, predicted_classes, average='macro') 
    scores = [round(precision, 4), round(recall, 4), round(fscore, 4)]

    return scores


def evaluate_model(img_root_dir, image_size, model, history, log_dir="", verbose_metrics=False):
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(os.path.join(img_root_dir, 'test'),
                                                class_mode='binary',
                                                color_mode='rgb',
                                                target_size=image_size,
                                                batch_size=1,
                                                shuffle=False,)

    # Calculate metrics results
    if verbose_metrics:
        _, acc, prec, rec, auc = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
        print('\nAccuracy on test set: %.3f' % (acc * 100.0))
        summarize_diagnostics_verbose(history, log_dir)
    else:
        _, acc = model.evaluate(test_generator, steps=len(test_generator), verbose=1)
        print('\nAccuracy on test set: %.3f' % (acc * 100.0))
        summarize_diagnostics_base(history, log_dir)

    # plot confusion matrix and results
    scores = show_results(model, test_generator, log_dir)
    
    return acc, scores


def evaluate_model_multi_view(img_root_dir, image_size, model, history, log_dir="", verbose_metrics=False, input_size=6):
    ppl_root_dir0 = f"{img_root_dir}_ppl_0"
    ppl_root_dir30 = f"{img_root_dir}_ppl_30"
    ppl_root_dir60 = f"{img_root_dir}_ppl_60"
    ppl_root_dir_merged = f"{img_root_dir}_ppl"

    xpl_root_dir0 = f"{img_root_dir}_xpl_0"
    xpl_root_dir30 = f"{img_root_dir}_xpl_30"
    xpl_root_dir60 = f"{img_root_dir}_xpl_60"
    xpl_root_dir_merged = img_root_dir
    
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    ppl_test_generator0 = test_datagen.flow_from_directory(os.path.join(ppl_root_dir0, 'test'),
                                                class_mode='binary',
                                                color_mode='rgb',
                                                target_size=image_size,
                                                batch_size=1,
                                                shuffle=False,)

    ppl_test_generator30 = test_datagen.flow_from_directory(os.path.join(ppl_root_dir30, 'test'),
                                                class_mode='binary',
                                                color_mode='rgb',
                                                target_size=image_size,
                                                batch_size=1,
                                                shuffle=False,)
    
    ppl_test_generator60 = test_datagen.flow_from_directory(os.path.join(ppl_root_dir60, 'test'),
                                                class_mode='binary',
                                                color_mode='rgb',
                                                target_size=image_size,
                                                batch_size=1,
                                                shuffle=False,)

    ppl_test_generator_merged = test_datagen.flow_from_directory(os.path.join(ppl_root_dir_merged, 'test'),
                                                class_mode='binary',
                                                color_mode='rgb',
                                                target_size=image_size,
                                                batch_size=1,
                                                shuffle=False,)
    
    xpl_test_generator0 = test_datagen.flow_from_directory(os.path.join(xpl_root_dir0, 'test'),
                                                class_mode='binary',
                                                color_mode='rgb',
                                                target_size=image_size,
                                                batch_size=1,
                                                shuffle=False,)

    
    xpl_test_generator30 = test_datagen.flow_from_directory(os.path.join(xpl_root_dir30, 'test'),
                                                class_mode='binary',
                                                color_mode='rgb',
                                                target_size=image_size,
                                                batch_size=1,
                                                shuffle=False,)
    
    xpl_test_generator60 = test_datagen.flow_from_directory(os.path.join(xpl_root_dir60, 'test'),
                                                class_mode='binary',
                                                color_mode='rgb',
                                                target_size=image_size,
                                                batch_size=1,
                                                shuffle=False,)

    xpl_test_generator_merged = test_datagen.flow_from_directory(os.path.join(xpl_root_dir_merged, 'test'),
                                                class_mode='binary',
                                                color_mode='rgb',
                                                target_size=image_size,
                                                batch_size=1,
                                                shuffle=False,)

    if input_size == 2:
        test_generator = joined_generator.JoinedGenerator2View(ppl_test_generator_merged, xpl_test_generator_merged)
    elif input_size == 3:
        test_generator = joined_generator.JoinedGenerator3View(xpl_test_generator0, xpl_test_generator30, xpl_test_generator60)
    else: # default: input_size == 6
        test_generator = joined_generator.JoinedGenerator6View(ppl_test_generator0, 
                                                               ppl_test_generator30, 
                                                               ppl_test_generator60, 
                                                               xpl_test_generator0, 
                                                               xpl_test_generator30, 
                                                               xpl_test_generator60)

    # Calculate metrics results
    if verbose_metrics:
        _, acc, prec, rec, auc = model.evaluate(test_generator, steps=len(ppl_test_generator0), verbose=1)
        print('\nAccuracy on test set: %.3f' % (acc * 100.0))
        summarize_diagnostics_verbose(history, log_dir)
    else:
        _, acc = model.evaluate(test_generator, steps=len(ppl_test_generator0), verbose=1)
        print('\nAccuracy on test set: %.3f' % (acc * 100.0))
        summarize_diagnostics_base(history, log_dir)

    # plot confusion matrix and results
    scores = show_results(model, test_generator, log_dir)
    
    return acc, scores


