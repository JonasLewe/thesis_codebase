import os
from utils import utils


def log_data(model, history, metrics, mean_iou_score):
    pass


def save_best_model(model, mean_iou_score, seed_value, PARENT_DIR):
    best_iou_score = utils.get_best_iou_score(PARENT_DIR)
    if mean_iou_score > best_iou_score:
        model.save(os.path.join(PARENT_DIR, f"Best_Model_seed={seed_value}_iou={mean_iou_score}"), overwrite=True)