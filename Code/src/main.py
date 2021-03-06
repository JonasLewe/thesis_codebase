# This project is run inside a custom conda environment, which is built using conda and pip together
# Since merely copying the environment via a env.yaml file does result in multiple issues,
# a clean manual install is recommended.
#
# On Windows follow these steps:
# - conda create -n "new_env" python=3.8
# - conda install tensorflow-gpu=2.3 tensorflow=2.3=mkl_py38h1fcfbd6_0
# - conda install <packages specified in ./requirements.yml>
# - pip install opencv-python (very important to install opencv via pip, conda install results in errors)
# - conda install <all other missing packages>:
#       - conda install -c conda-forge wandb
#       - conda install scikit-image
#       - conda install openpyxl
#
# On Windows systems it might be necessary to enable long paths via
# CMD -> regedit -> HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem -> LongPathsEnabled -> 1 
#
# On linux systems the required steps might be different and your mileage may vary...

import os
import sys
import shutil
# from cv2 import threshold
import yaml
from operator import itemgetter
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from timeit import default_timer as timer

# disable cuda debug info
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

root_dir = os.path.join("..", "..")
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
# root_image_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["input_data_undersampled"])

# test non-merged xpl data
# root_image_folder = f"{root_image_folder}_xpl_0"

# test merged ppl data
# root_image_folder = f"{root_image_folder}_ppl"



class_0_img_folder_ppl_merged = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_0_img_folder_ppl_merged"])
class_0_img_folder_ppl_0 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_0_img_folder_ppl_0"])
class_0_img_folder_ppl_30 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_0_img_folder_ppl_30"])
class_0_img_folder_ppl_60 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_0_img_folder_ppl_60"])
class_0_img_folder_xpl_merged = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_0_img_folder_xpl_merged"])
class_0_img_folder_xpl_0 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_0_img_folder_xpl_0"])
class_0_img_folder_xpl_30 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_0_img_folder_xpl_30"])
class_0_img_folder_xpl_60 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_0_img_folder_xpl_60"])
class_1_img_folder_ppl_merged = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_1_img_folder_ppl_merged"])
class_1_img_folder_ppl_0 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_1_img_folder_ppl_0"])
class_1_img_folder_ppl_30 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_1_img_folder_ppl_30"])
class_1_img_folder_ppl_60 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_1_img_folder_ppl_60"])
class_1_img_folder_xpl_merged = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_1_img_folder_xpl_merged"])
class_1_img_folder_xpl_0 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_1_img_folder_xpl_0"])
class_1_img_folder_xpl_30 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_1_img_folder_xpl_30"])
class_1_img_folder_xpl_60 = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["class_1_img_folder_xpl_60"])

polygon_label_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["polygon_label_folder"])
voc_label_folder = os.path.join(ROOT_DIR, base_config[user_config["dataset"]]["voc_label_folder"])
last_conv_layer_name = base_config[user_config["model_type"]]["last_conv_layer_name"]
image_size = (base_config["global"]["img_size"], base_config["global"]["img_size"])
max_seed_value = base_config["global"]["max_seed_value"]
single_run = base_config["global"]["single_run"]
xlsx_results_filename = base_config["xlsx_results_filepath"]
xlsx_input_split_filename = base_config["xlsx_input_split_filepath"]


# Read varlables from user config
model_type = user_config["model_type"]
dataset = user_config["dataset"]
epochs = user_config["epochs"]
batch_size = user_config["batch_size"]
learning_rate = user_config["learning_rate"]
iou_threshold = user_config["iou_threshold"]
use_gpu = user_config["use_gpu"]
verbose_metrics = user_config["verbose_metrics"]
early_fusion = user_config["early_fusion"]
fusion_technique = user_config["fusion_technique"]
input_size = user_config["input_size"]
dropout_value = user_config["dropout_value"]
regularization = user_config["regularization"]
num_hidden_layers = user_config["num_hidden_layers"]


if not use_gpu:
    # disable GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Only import tensorflow after setting environmant variables!
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import wandb
from wandb.keras import WandbCallback
from ml_base import evaluation, train, seeds, metrics 
from log_data import logging, xlsx
from utils import utils

now = datetime.now()
current_time = now.strftime("%d_%m_%Y-%H_%M_%S")
PARENT_DIR = os.path.join(ROOT_DIR, "Data", "Output_Data", "model_tuning", user_config["parent_dir"])
run_dir_name = f"{user_config['dataset']}_{user_config['model_type']}_{current_time}"
RUN_DIR = os.path.join(PARENT_DIR, run_dir_name) 
XLSX_RESULTS_FILE = os.path.join(ROOT_DIR, "Data", "Output_Data", xlsx_results_filename)
XLSX_INPUT_SPLIT_FILE = os.path.join(ROOT_DIR, "Config", "xlsx", xlsx_input_split_filename)
comments = args.comment

# initialize weights and biases
# wandb.init(project="olivine_classifier", entity="089jonas", group="evaluating_seeds")
# 
# # configure weights and biases callback
# wandb.config = {
#     "learning_rate": learning_rate,
#     "epochs": epochs,
#     "batch_size": batch_size
# }


if __name__=="__main__":
    # create parent folder here...
    Path(PARENT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RUN_DIR).mkdir(parents=True, exist_ok=False)
    print("\nNum GPUs Available:", len(tf.config.experimental.list_physical_devices("GPU")))

    # gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    # session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


    # copy used settings to current folder
    shutil.copy(os.path.join(ROOT_DIR, CONFIG, args.config), os.path.join(RUN_DIR, "settings.yml"))
    xlsx_summary = []

    start_seed = 1
    if single_run:
        start_seed = max_seed_value

    for seed_value in range(start_seed, max_seed_value+1):
        start = timer()
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
                tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                 min_delta=0,
                                                 patience=45,
                                                 verbose=1,
                                                 mode="max",
                                                 baseline=None,
                                                 restore_best_weights=True,
                                                )
                # WandbCallback()
            ]
            
            # if model is a single-view model
            if input_size == 1:
                model, history = train.train_model_single_view(root_image_folder,
                                                image_size,
                                                callbacks=callbacks,
                                                verbose_metrics=verbose_metrics,
                                                model_name=user_config["model_type"],
                                                epochs=epochs,
                                                batch_size=batch_size,
                                                learning_rate=learning_rate,
                                                dropout=dropout_value,
                                                regularization=regularization,
                                                num_hidden_layers=num_hidden_layers
                                                )

                accuracy, scores = evaluation.evaluate_model(root_image_folder,
                                                        image_size,
                                                        model,
                                                        history,
                                                        log_dir=SUB_DIR,
                                                        verbose_metrics=verbose_metrics,
                                                        )

            # if model is a multi-view model
            else:
                if dataset == "olivine":
                    model, history = train.train_model_multi_view(root_image_folder,
                                                    image_size,
                                                    callbacks=callbacks,
                                                    verbose_metrics=verbose_metrics,
                                                    model_name=user_config["model_type"],
                                                    early_fusion=early_fusion,
                                                    fusion_technique=fusion_technique,
                                                    input_size=input_size,
                                                    epochs=epochs,
                                                    batch_size=batch_size,
                                                    learning_rate=learning_rate,
                                                    dropout=dropout_value,
                                                    regularization=regularization,
                                                    num_hidden_layers=num_hidden_layers
                                                    )

                    accuracy, scores = evaluation.evaluate_model_multi_view(root_image_folder,
                                                            image_size,
                                                            model,
                                                            history,
                                                            log_dir=SUB_DIR,
                                                            verbose_metrics=verbose_metrics,
                                                            input_size=input_size
                                                            )
                else:
                    # if dataset == "cats_vs_dogs"
                    model, history = train.train_model_multi_view_cats_vs_dogs(root_image_folder,
                                                        image_size,
                                                        callbacks=callbacks,
                                                        verbose_metrics=verbose_metrics,
                                                        model_name=user_config["model_type"],
                                                        early_fusion=early_fusion,
                                                        fusion_technique=fusion_technique,
                                                        input_size=input_size,
                                                        epochs=epochs,
                                                        batch_size=batch_size,
                                                        learning_rate=learning_rate,
                                                        dropout=dropout_value,
                                                        regularization=regularization,
                                                        num_hidden_layers=num_hidden_layers
                                                        )

                    accuracy, scores = evaluation.evaluate_model_multi_view_cats_vs_dogs(root_image_folder,
                                                            image_size,
                                                            model,
                                                            history,
                                                            log_dir=SUB_DIR,
                                                            verbose_metrics=verbose_metrics,
                                                            input_size=input_size
                                                            )

            # check if dataset supports segmentation 
            if (class_1_img_folder_xpl_merged != ""):
                if input_size == 1: # single view is selected
                    segmentation_scores = metrics.calc_mean_segmentation_scores_single_input(
                                                                class_0_img_folder_xpl_merged,
                                                                class_1_img_folder_xpl_merged,
                                                                image_size, 
                                                                model, 
                                                                polygon_label_folder, 
                                                                voc_label_folder, 
                                                                last_conv_layer_name, 
                                                                iou_threshold,
                                                                cam_img_output_path=GRAD_CAM_IMGS_DIR,
                                                                xlsx_input_split_file=XLSX_INPUT_SPLIT_FILE,
                                                                )
                else: # multi-view is selected
                    segmentation_scores = metrics.calc_mean_segmentation_scores_multi_input(
                                                                class_0_img_folder_ppl_0,
                                                                class_0_img_folder_ppl_30,
                                                                class_0_img_folder_ppl_60,
                                                                class_0_img_folder_ppl_merged,
                                                                class_0_img_folder_xpl_0,
                                                                class_0_img_folder_xpl_30,
                                                                class_0_img_folder_xpl_60,
                                                                class_0_img_folder_xpl_merged,
                                                                class_1_img_folder_ppl_0,
                                                                class_1_img_folder_ppl_30,
                                                                class_1_img_folder_ppl_60,
                                                                class_1_img_folder_ppl_merged,
                                                                class_1_img_folder_xpl_0,
                                                                class_1_img_folder_xpl_30,
                                                                class_1_img_folder_xpl_60,
                                                                class_1_img_folder_xpl_merged,
                                                                input_size,
                                                                image_size, 
                                                                model, 
                                                                polygon_label_folder, 
                                                                voc_label_folder, 
                                                                last_conv_layer_name, 
                                                                iou_threshold,
                                                                cam_img_output_path=GRAD_CAM_IMGS_DIR,
                                                                xlsx_input_split_file=XLSX_INPUT_SPLIT_FILE,
                                                                )

                mean_iou_score = segmentation_scores[0]
                mean_iou_olivine_score = segmentation_scores[1]
                max_iou_olivine_score = segmentation_scores[2]
                min_iou_olivine_score = segmentation_scores[3]
                xlsx_summary.append((round(mean_iou_score, 4),
                                     round(mean_iou_olivine_score, 4),
                                     round(max_iou_olivine_score, 4),
                                     round(min_iou_olivine_score, 4),
                                     seed_value,
                                     scores))
                # logging.log_data(model, history, accuracy, mean_iou_score)
                logging.save_best_model(model, segmentation_scores, seed_value, RUN_DIR)

        except BaseException as error:
            print(f"Error: {error}")
            # remove all folders for current run
            shutil.rmtree(RUN_DIR)

        # calculate time elapsed        
        stop = timer()
        elapsed_time = stop - start
        utils.display_wall_clock_time(elapsed_time, max_seed_value, seed_value) 

    # check if dataset supports segmentation 
    if (class_1_img_folder_xpl_merged != ""):
        # get results from best round
        mean_iou_score = max(xlsx_summary, key=itemgetter(0))[0]
        mean_iou_olivine = max(xlsx_summary, key=itemgetter(0))[1]
        max_iou_olivine = max(xlsx_summary, key=itemgetter(0))[2]
        min_iou_olivine = max(xlsx_summary, key=itemgetter(0))[3]
        max_iou_seed = max(xlsx_summary, key=itemgetter(0))[4]
        max_iou_precision = max(xlsx_summary, key=itemgetter(0))[5][0]
        max_iou_recall = max(xlsx_summary, key=itemgetter(0))[5][1]
        max_iou_fscore = max(xlsx_summary, key=itemgetter(0))[5][2]

        # save results of best run to xlsx file
        xlsx.parse_model_output_to_xlsx(run_dir_name, 
                                        user_config["model_type"],
                                        input_size, 
                                        epochs, 
                                        learning_rate, 
                                        batch_size, 
                                        iou_threshold,
                                        dropout_value,
                                        regularization,
                                        max_iou_seed,
                                        max_seed_value,
                                        mean_iou_score,
                                        mean_iou_olivine,
                                        max_iou_olivine,
                                        min_iou_olivine,
                                        max_iou_precision,
                                        max_iou_recall,
                                        max_iou_fscore,
                                        use_gpu,
                                        comments,
                                        XLSX_RESULTS_FILE,
                                        )