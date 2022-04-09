import os
import shutil
import yaml
from datetime import datetime
from argparse import ArgumentParser

# disable cuda debug info
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

ROOT_DIR = "../../"

parser = ArgumentParser()
# parser.add_argument("-d", "--dataset", type=str, default="olivine",
#                     help="Datasets available: "olivine", "cats_vs_dogs"")
# parser.add_argument("-m", "--model_type", type=str, default="base",
#                     help="Model types available: "base", "vgg"")
# parser.add_argument("-c", "--cam_save", type=int, default=0,
#                     help="Specify if cam images should be saved.")
# parser.add_argument("-tp", "--tuning_parameter", type=str, default="model",
#                     help="Specify if cam images should be saved.")
parser.add_argument("-c", "--config", type=str, default="dummy.yml",
                    help="Specify the config file to be used.")
args = parser.parse_args()

# load custom config data
with open(os.path.join(ROOT_DIR, args.config), "rb") as f:
    user_config = yaml.safe_load(f)

# load base config data
with open(os.path.join(ROOT_DIR, "base_config.yml"), "rb") as f:
    base_config = yaml.safe_load(f)

root_image_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["input_data"])
class_1_img_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_1_img_folder"])
polygon_label_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["polygon_label_folder"])
voc_label_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["voc_label_folder"])
last_conv_layer_name = os.path.join(ROOT_DIR, base_config[user_config["model_type"]]["last_conv_layer_name"])
image_size = (base_config["global"]["img_size"], base_config["global"]["img_size"])
max_seed_value = base_config["max_seed_value"]

epochs = user_config["epochs"]
use_gpu = user_config["use_gpu"]


gpu_status = "gpu"
if not use_gpu:
    # disable GPU
    gpu_status = "nogpu"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Only import tensorflow after setting environmant variables!
import tensorflow as tf
from ml_base import evaluation, train, seeds, metrics
from logging import logging

now = datetime.now()
current_time = now.strftime("%d_%m_%Y-%H_%M_%S")
# LOG_DIR = os.path.join(ROOT_DIR, "Data", "Output_Data", "tensorboard", "logs", f"{user_config["dataset"]}_{gpu_status}_seed{max_seed_value}_{current_time}")
PARENT_DIR = os.path.join(ROOT_DIR, "Data", "Output_Data", "model_tuning", user_config["parent_dir"])


if __name__=="__main__":
    # create parent folder here...
    os.makedirs(PARENT_DIR)
    print("\nNum GPUs Available:", len(tf.config.experimental.list_physical_devices("GPU")))
    for seed_value in range(1, max_seed_value): 
        try:
            # create folders for every run
            SUB_DIR = os.path.join(PARENT_DIR, f"seed={seed_value}")
            TENSORBOARD_DIR = os.path.join(SUB_DIR, "tensorboard")
            GRAD_CAM_IMGS_DIR = os.path.join(SUB_DIR, "grad_cam_imgs")
            os.makedirs(SUB_DIR)    
            os.makedirs(TENSORBOARD_DIR)   
            os.makedirs(GRAD_CAM_IMGS_DIR)

            # fix all random seeds according to keras documentation
            seeds.fix_seeds_keras_style(seed_value)
            
            # set callbacks for training
            callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, profile_batch=0)
            ]

            model, history = train.train_model(root_image_folder, image_size, callbacks=callbacks, model_name=args.model_type, epochs=epochs)
            accuracy = evaluation.evaluate_model(root_image_folder, image_size, model, history, verbose=True, log_dir=SUB_DIR)
            mean_iou_score = metrics.calc_mean_iou_score(class_1_img_folder,
                                                         image_size, 
                                                         model, 
                                                         polygon_label_folder, 
                                                         voc_label_folder, 
                                                         last_conv_layer_name, 
                                                         cam_img_output_path=GRAD_CAM_IMGS_DIR)
            logging.log_data(model, history, accuracy, mean_iou_score)
            logging.save_best_model(model, PARENT_DIR)
            print(f"Mean IoU score: {mean_iou_score}")
        except BaseException as error:
            print(f"Error: {error}")
            shutil.rmtree(SUB_DIR)
            shutil.rmtree(TENSORBOARD_DIR)
            shutil.rmtree(GRAD_CAM_IMGS_DIR)