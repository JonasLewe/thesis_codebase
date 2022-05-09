import os
import sys
import shutil
# from cv2 import threshold
import yaml
from operator import itemgetter
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

# disable cuda debug info
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

root_dir = "../../"
ROOT_DIR = os.path.join(os.path.dirname(__file__), root_dir)
# ROOT_DIR = os.path.dirname(__file__)
CONFIG = "Config"


parser = ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="user_config.yml",
                    help="Specify the config file to be used.")
parser.add_argument("--comment", type=str, default="",
                    help="Add comments to xlsx entry.")
args = parser.parse_args()

# load custom config data
with open(os.path.join(ROOT_DIR, CONFIG, args.config), "rb") as f:
    user_config = yaml.safe_load(f)

# load base config data
with open(os.path.join(ROOT_DIR, CONFIG, "base_config.yml"), "rb") as f:
    base_config = yaml.safe_load(f)

# Read variables from base config
root_image_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["input_data"])
# root_image_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["input_data_small"])
class_1_img_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_1_img_folder"])
class_0_img_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_0_img_folder"])
polygon_label_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["polygon_label_folder"])
voc_label_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["voc_label_folder"])
last_conv_layer_name = base_config[user_config["model_type"]]["last_conv_layer_name"]
image_size = (base_config["global"]["img_size"], base_config["global"]["img_size"])
max_seed_value = base_config["global"]["max_seed_value"]
xlsx_results_filename = base_config["xlsx_results_filepath"]
xlsx_input_split_filename = base_config["xlsx_input_split_filepath"]

# Read varlables from user config
epochs = user_config["epochs"]
batch_size = user_config["batch_size"]
learning_rate = user_config["learning_rate"]
iou_threshold = user_config["iou_threshold"]
use_gpu = user_config["use_gpu"]
verbose_metrics = user_config["verbose_metrics"]


if not use_gpu:
    # disable GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Only import tensorflow after setting environmant variables!
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from ml_base import evaluation, train, seeds, metrics
from log_data import logging, xlsx

now = datetime.now()
current_time = now.strftime("%d_%m_%Y-%H_%M_%S")
PARENT_DIR = os.path.join(ROOT_DIR, "Data", "Output_Data", "model_tuning", user_config["parent_dir"])
run_dir_name = f"{user_config['dataset']}_{user_config['model_type']}_{current_time}"
RUN_DIR = os.path.join(PARENT_DIR, run_dir_name) 
XLSX_RESULTS_FILE = os.path.join(ROOT_DIR, "Data", "Output_Data", xlsx_results_filename)
XLSX_INPUT_SPLIT_FILE = os.path.join(ROOT_DIR, "Config", "xlsx", xlsx_input_split_filename)
comments = args.comment

# initialize weights and biases
wandb.init(project="olivine_classifier", entity="089jonas", group="evaluating_seeds")

# configure weights and biases callback
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size
}


if __name__=="__main__":
    # create parent folder here...
    Path(PARENT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RUN_DIR).mkdir(parents=True, exist_ok=False)
    print("\nNum GPUs Available:", len(tf.config.experimental.list_physical_devices("GPU")))

    # copy used settings to current folder
    shutil.copy(os.path.join(ROOT_DIR, CONFIG, args.config), os.path.join(RUN_DIR, "settings.yml"))
    xlsx_summary = []

    for seed_value in range(1, max_seed_value+1): 
        print(53*"#")
        print(f"Run {seed_value}/{max_seed_value}:")
        print(53*"#")
        try:
            # create folders for every run
            SUB_DIR = os.path.join(RUN_DIR, f"seed={seed_value}")
            TENSORBOARD_DIR = os.path.join(SUB_DIR, "tensorboard")
            GRAD_CAM_IMGS_DIR = os.path.join(SUB_DIR, "grad_cam_imgs")
            os.makedirs(SUB_DIR)    
            os.makedirs(TENSORBOARD_DIR)   
            os.makedirs(GRAD_CAM_IMGS_DIR)

            # fix all random seeds according to keras documentation
            seeds.fix_seeds_keras_style(seed_value)
            
            # set callbacks for training
            callbacks = [
                tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, profile_batch=0),
                WandbCallback()
            ]

            model, history = train.train_model(root_image_folder, image_size, callbacks=callbacks, verbose_metrics=verbose_metrics, model_name=user_config["model_type"], epochs=epochs, batch_size=batch_size)
            accuracy, scores = evaluation.evaluate_model(root_image_folder, image_size, model, history, log_dir=SUB_DIR, verbose_metrics=verbose_metrics)

            # check if dataset supports segmentation 
            if (class_1_img_folder is not ""):
                mean_iou_score = metrics.calc_mean_iou_score(class_1_img_folder,
                                                             class_0_img_folder,
                                                             image_size, 
                                                             model, 
                                                             polygon_label_folder, 
                                                             voc_label_folder, 
                                                             last_conv_layer_name, 
                                                             iou_threshold,
                                                             cam_img_output_path=GRAD_CAM_IMGS_DIR,
                                                             xlsx_input_split_file=XLSX_INPUT_SPLIT_FILE
                                                             )

                xlsx_summary.append((round(mean_iou_score, 4), seed_value, scores)) 
                # logging.log_data(model, history, accuracy, mean_iou_score)
                logging.save_best_model(model, mean_iou_score, seed_value, RUN_DIR)

                # get results from best round
                max_iou_score = max(xlsx_summary, key=itemgetter(0))[0]
                max_iou_seed = max(xlsx_summary, key=itemgetter(0))[1]
                max_iou_precision = max(xlsx_summary, key=itemgetter(0))[2][0]
                max_iou_recall = max(xlsx_summary, key=itemgetter(0))[2][1]
                max_iou_fscore = max(xlsx_summary, key=itemgetter(0))[2][2]

                xlsx.parse_model_output_to_xlsx(run_dir_name, 
                                                user_config["model_type"], 
                                                epochs, 
                                                learning_rate, 
                                                batch_size, 
                                                iou_threshold,
                                                max_iou_seed,
                                                max_seed_value,
                                                max_iou_score,
                                                max_iou_precision,
                                                max_iou_recall,
                                                max_iou_fscore,
                                                use_gpu,
                                                comments,
                                                XLSX_RESULTS_FILE)
        except BaseException as error:
            print(f"Error: {error}")
            # remove all folders for current run
            shutil.rmtree(RUN_DIR)
