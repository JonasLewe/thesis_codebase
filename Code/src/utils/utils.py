import os
import re
import numpy as np


def calc_mean(img_tensor_list):
    return np.mean(img_tensor_list, axis=0)


def calc_std(img_tensor_list):
    return np.std(img_tensor_list, axis=0)


def get_best_iou_score(subdir):
    best_model_pattern = "Best_Model_seed=\d_iou=\d.\d+.h5"
    pattern = re.compile(best_model_pattern)
    for filepath in os.listdir(subdir):
        if pattern.match(filepath):
            best_iou_score = float(filepath.split("_")[-1].split(".h5")[0].split("=")[-1])
        else:
            best_iou_score = 0
    return best_iou_score