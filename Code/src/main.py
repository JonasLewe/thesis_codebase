import os
import shutil
# from cv2 import mean
import yaml
from datetime import datetime
from argparse import ArgumentParser

# disable cuda debug info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ROOT_DIR = '../../'

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default='olivine',
                    help="Datasets available: 'olivine', 'cats_vs_dogs'")
parser.add_argument("-m", "--model_type", type=str, default='base',
                    help="Model types available: 'base', 'vgg'")
parser.add_argument("-c", "--cam_save", type=int, default=0,
                    help="Specify if cam images should be saved.")
args = parser.parse_args()

# load base config data
with open(os.path.join(ROOT_DIR, 'base_config.yml'), "rb") as f:
    config = yaml.safe_load(f)

root_image_folder = os.path.join(ROOT_DIR, config[args.dataset]["input_data"])
class_1_img_folder = os.path.join(ROOT_DIR, config[args.dataset]["class_1_img_folder"])
polygon_label_folder = os.path.join(ROOT_DIR, config[args.dataset]["polygon_label_folder"])
voc_label_folder = os.path.join(ROOT_DIR, config[args.dataset]["voc_label_folder"])
last_conv_layer_name = os.path.join(ROOT_DIR, config[args.model_type]["last_conv_layer_name"])
image_size = (config["global"]["img_size"], config["global"]["img_size"])
epochs = config["global"]["epochs"]
use_gpu = config["global"]["use_gpu"]
max_seed_value = config["global"]["max_seed_value"]


gpu_status = 'gpu'
if not use_gpu:
    # disable GPU
    gpu_status = 'nogpu'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Only import tensorflow after setting environmant variables!
import tensorflow as tf
from ml_base import evaluation, train, seeds, metrics
from logging import logging

now = datetime.now()
current_time = now.strftime("%d_%m_%Y-%H_%M_%S")
# LOG_DIR = os.path.join(ROOT_DIR, 'Data', 'Output_Data', 'tensorboard', 'logs', f"{args.dataset}_{gpu_status}_seed{max_seed_value}_{current_time}")
PARENT_LOG_DIR = os.path.join(ROOT_DIR, 'Data', 'Output_Data', 'model_tuning')
os.makedirs(LOG_DIR)

if args.cam_save:
    cam_img_output_path = LOG_DIR
else:
    cam_img_output_path = ''

if __name__=='__main__':
    # create parent folder here...
    print("\nNum GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
    try:
        for seed_value in range(1, max_seed_value): 
            # create folder for every run

            # fix all random seeds according to keras documentation
            seeds.fix_seeds_keras_style(seed_value)
            callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, profile_batch=0)
            ]

            model, history = train.train_model(root_image_folder, image_size, callbacks=callbacks, model_name=args.model_type, epochs=epochs)
            accuracy = evaluation.evaluate_model(root_image_folder, image_size, model, history, verbose=True, log_dir=LOG_DIR)
            mean_iou_score = metrics.calc_mean_iou_score(class_1_img_folder,
                                                        image_size, 
                                                        model, 
                                                        polygon_label_folder, 
                                                        voc_label_folder, 
                                                        last_conv_layer_name, 
                                                        cam_img_output_path)
            logging.log_data(model, history, accuracy, mean_iou_score)
            print(f"Mean IoU score: {mean_iou_score}")
    except BaseException as error:
        print(f"Error: {error}")
        shutil.rmtree(LOG_DIR)