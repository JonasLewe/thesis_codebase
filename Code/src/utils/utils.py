import os
import numpy as np
from openpyxl import load_workbook


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


def get_val_test_data(xlsx_file):
    wb = load_workbook(xlsx_file)
    ws_test = wb["Test"]
    ws_val = wb ["Validation"]
    
    valset = []
    testset = []

    for row in ws_val:
        file_id = "_".join([row[1].value,row[2].value])
        valset.append(file_id)

    for row in ws_test:
        file_id = "_".join([row[1].value,row[2].value])
        testset.append(file_id)

    return valset, testset


def display_wall_clock_time(elapsed_time, max_iterations, current_iteration):
    unit = "seconds"
    print(53*"#")
    print(f"Run #{current_iteration} time: {round(elapsed_time, 2)} {unit}.")
    remaining_time = round((max_iterations - current_iteration) * elapsed_time, 2)
    if remaining_time > 60:
        remaining_time = round(remaining_time / 60, 2)
        unit = "minutes"
    print(f"Approximate remaining time: {remaining_time} {unit}.")
    print(53*"#")