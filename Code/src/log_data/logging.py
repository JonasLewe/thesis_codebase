import re
import os
from utils import utils


def log_data(model, history, metrics, mean_iou_score):
    pass


def save_best_model(model, mean_iou_score, seed_value, PARENT_DIR):
    best_iou_score = utils.get_best_iou_score(PARENT_DIR)
    best_model_pattern = "Best_Model_seed=\d_iou=\d.\d+.h5"
    pattern = re.compile(best_model_pattern)
    if mean_iou_score > best_iou_score:
        for filepath in os.listdir(PARENT_DIR):
            if pattern.match(filepath):
                os.remove(os.path.join(PARENT_DIR, filepath))
        model.save(os.path.join(PARENT_DIR, f"Best_Model_seed={seed_value}_mean_iou={mean_iou_score:.4f}.h5"), overwrite=True)