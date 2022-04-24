import os
import numpy as np
from skimage.transform import resize
from ml_base.grad_CAM import cam_pipeline 
from utils.img import get_json_img_name, draw_json_polygons
import tensorflow as tf


def calc_iou_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


# def calc_mean_iou_score(class_1_img_folder, IMAGE_SIZE, model, polygon_label_folder, voc_label_folder, last_conv_layer_name, cam_img_output_path, threshold=0.01):
#     iou_scores = []
#     for json_file_name in os.listdir(polygon_label_folder):
#         img_name = get_json_img_name(json_file_name)
#         image_id = "_".join(img_name.split('_')[:3])
#         json_img = draw_json_polygons(img_name, json_file_name, class_1_img_folder, polygon_label_folder, IMAGE_SIZE)
#         pred_heatmap = cam_pipeline(class_1_img_folder, img_name, json_img, IMAGE_SIZE, model, last_conv_layer_name, cam_img_output_path, draw_text=True)
#         predicted_binary_heatmap = np.where(pred_heatmap > threshold, 1, 0)
#         heatmap_size = predicted_binary_heatmap.shape
# 
#         # search for corresponding labeled .npy file:
#         for file in os.listdir(voc_label_folder):
#             if image_id in file:
#                 ground_truth_heatmap = np.load(os.path.join(voc_label_folder, file))
#                 ground_truth_heatmap = resize(ground_truth_heatmap, heatmap_size)
#                 iou_score = calc_iou_score(ground_truth_heatmap, predicted_binary_heatmap)
#                 iou_scores.append(iou_score)
# 
#     mean_iou = np.mean(iou_scores)
#     return mean_iou


def calc_mean_iou_score(class_1_img_folder, IMAGE_SIZE, model, polygon_label_folder, voc_label_folder, last_conv_layer_name, iou_threshold, cam_img_output_path):
    predicted_binary_added_heatmaps = np.zeros([0])
    ground_truth_added_heatmaps = np.zeros([0])

    # predict heatmaps for olivine images
    for json_file_name in os.listdir(polygon_label_folder):
        img_name = get_json_img_name(json_file_name)
        image_id = "_".join(img_name.split('_')[:3])
        segmented_img = draw_json_polygons(img_name, json_file_name, class_1_img_folder, polygon_label_folder, IMAGE_SIZE)
        pred_heatmap = cam_pipeline(class_1_img_folder, img_name, segmented_img, IMAGE_SIZE, model, last_conv_layer_name, cam_img_output_path, draw_text=True)
        predicted_binary_heatmap = np.where(pred_heatmap > iou_threshold, 1, 0)
        heatmap_size = predicted_binary_heatmap.shape
        
        #add all predicted segmentations to one large vector
        predicted_binary_added_heatmaps = np.concatenate((predicted_binary_added_heatmaps, predicted_binary_heatmap.flatten()), axis=None)

        # search for corresponding labeled .npy file:
        for file in os.listdir(voc_label_folder):
            if image_id in file:
                ground_truth_heatmap = np.load(os.path.join(voc_label_folder, file))
                ground_truth_heatmap = resize(ground_truth_heatmap, heatmap_size)

                # add all ground truth segmentations to one large 1D vector
                ground_truth_added_heatmaps = np.concatenate((ground_truth_added_heatmaps, ground_truth_heatmap.flatten()), axis=None)

    # predict heatmaps for non-olivine images
    # TODO
    # for file in os.listdir(non_olivine_folder):
    #   if 

    # calculating the iou value over all images combined 
    mean_iou = calc_iou_score(ground_truth_added_heatmaps, predicted_binary_added_heatmaps)

    return mean_iou

METRICS = [
     # tf.keras.metrics.TruePositives(name='tp'),
     # tf.keras.metrics.FalsePositives(name='fp'),
     # tf.keras.metrics.TrueNegatives(name='tn'),
     # tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]