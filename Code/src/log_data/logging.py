# import re
import os
from utils import utils


def log_data(model, history, metrics, mean_iou_score):
    pass


def save_best_model(model, mean_iou_score, seed_value, PARENT_DIR):
    print(f"Mean IoU score: {mean_iou_score:.4f}")
    best_iou_score = utils.get_best_iou_score(PARENT_DIR)
    # best_model_pattern = "Best_Model_seed=\d_mean_iou=0.\d+.h5"
    # pattern = re.compile(best_model_pattern)
    if mean_iou_score > best_iou_score:
        print(f"Current best iou score: {best_iou_score}\nNew best iou score: {mean_iou_score:.4f}")
        for filepath in os.listdir(PARENT_DIR):
            # if pattern.match(filepath):
            if filepath.endswith(".h5"):
                os.remove(os.path.join(PARENT_DIR, filepath))
        model.save(os.path.join(PARENT_DIR, f"Best_Model_seed={seed_value}_mean_iou={mean_iou_score:.4f}.h5"), overwrite=True)
        print(f"Best IoU score updated.")