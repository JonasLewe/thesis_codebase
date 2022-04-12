import os
import numpy as np


def calc_mean(img_tensor_list):
    return np.mean(img_tensor_list, axis=0)


def calc_std(img_tensor_list):
    return np.std(img_tensor_list, axis=0)


def get_best_iou_score(subdir):
    for filepath in os.listdir(subdir):
        if filepath.endswith(".h5"):
            best_iou_score = float(filepath.split("_")[-1].split("=")[1].split(".h5")[0])
            break
        else:
            best_iou_score = 0
    return best_iou_score