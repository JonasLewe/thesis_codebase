import os
import numpy as np
from skimage.transform import resize
from sklearn.metrics import jaccard_score
from ml_base import grad_CAM
from utils.img import get_json_img_name, draw_json_polygons
from utils.utils import get_val_test_data
import tensorflow as tf


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


def calc_iou_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Legacy method of calculating IoU by calculating the IoU of every image and the average afterwards 
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


def calc_mean_segmentation_scores_single_input(class_0_img_folder, class_1_img_folder, image_size, model, polygon_label_folder, voc_label_folder, last_conv_layer_name, iou_threshold, cam_img_output_path, xlsx_input_split_file):
    print("Calculating segmentation scores...")
    valset, testset = get_val_test_data(xlsx_input_split_file)
    predicted_binary_added_heatmaps = np.zeros([0])
    ground_truth_added_heatmaps = np.zeros([0])

    predicted_binary_added_heatmaps_olivine = np.zeros([0])
    ground_truth_added_heatmaps_olivine = np.zeros([0])

    predicted_binary_added_heatmaps_non_olivine = np.zeros([0])
    ground_truth_added_heatmaps_non_olivine = np.zeros([0])

    olivine_iou_list = []
    non_olivine_iou_list = []

    # predict heatmaps for olivine images
    for json_file_name in os.listdir(polygon_label_folder):
        img_name = get_json_img_name(json_file_name)
        image_id = "_".join(img_name.split('_')[:3])
        segmented_img = draw_json_polygons(img_name, json_file_name, class_1_img_folder, polygon_label_folder, image_size)
        pred_heatmap = grad_CAM.cam_pipeline(class_1_img_folder, img_name, image_size, model, last_conv_layer_name, cam_img_output_path=cam_img_output_path, segmented_img=segmented_img)
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
        olivine_iou = calc_iou_score(ground_truth_heatmap, pred_heatmap)
        olivine_iou_list.append(olivine_iou)


    # calculate IoU score for olivine images only
    olivine_mean_iou = calc_iou_score(ground_truth_added_heatmaps, predicted_binary_added_heatmaps)
    # olivine_mean_dice = jaccard_score(ground_truth_added_heatmaps, predicted_binary_added_heatmaps)


    # predict heatmaps for non-olivine images in validation and test set and add to existing heatmaps
    for file in os.listdir(class_0_img_folder):
        # check if file is in val/test set
        for id in [*valset, *testset]:
            if id in file:
                pred_heatmap = grad_CAM.cam_pipeline(class_0_img_folder, file, image_size, model, last_conv_layer_name, draw_text=False)
                predicted_binary_heatmap = np.where(pred_heatmap > iou_threshold, 1, 0)
                # create empty ground truth heatmap, since it is expected to not detect any olivine in non-olivine images
                ground_truth_heatmap = np.zeros(predicted_binary_heatmap.shape) 
                predicted_binary_added_heatmaps = np.concatenate((predicted_binary_added_heatmaps, predicted_binary_heatmap.flatten()), axis=None)
                ground_truth_added_heatmaps = np.concatenate((ground_truth_added_heatmaps, ground_truth_heatmap.flatten()), axis=None)


    # calculating the iou value over all images combined 
    global_mean_iou = calc_iou_score(ground_truth_added_heatmaps, predicted_binary_added_heatmaps)
    # global_mean_dice = jaccard_score(ground_truth_added_heatmaps, predicted_binary_added_heatmaps)

    max_olivine_iou = np.max(olivine_iou_list)
    min_olivine_iou = np.min(olivine_iou_list)

    return [global_mean_iou, olivine_mean_iou, max_olivine_iou, min_olivine_iou]


def calc_mean_segmentation_scores_dual_input(class_0_img_folder_ppl, 
                                             class_0_img_folder_xpl,
                                             class_1_img_folder_ppl,
                                             class_1_img_folder_xpl, 
                                             image_size, 
                                             model, 
                                             polygon_label_folder, 
                                             voc_label_folder, 
                                             last_conv_layer_name, 
                                             iou_threshold, 
                                             cam_img_output_path, 
                                             xlsx_input_split_file):
    print("Calculating IoU score...")
    base_img_folders_class_0 = [class_0_img_folder_ppl, class_0_img_folder_xpl]
    base_img_folders_class_1 = [class_1_img_folder_ppl, class_1_img_folder_xpl]
    valset, testset = get_val_test_data(xlsx_input_split_file)
    predicted_binary_added_heatmaps = np.zeros([0])
    ground_truth_added_heatmaps = np.zeros([0])

    olivine_iou_list = []

    # predict heatmaps for olivine images
    for json_file_name in os.listdir(polygon_label_folder):
        img_name = get_json_img_name(json_file_name)
        image_id = "_".join(img_name.split('_')[:3])
        segmented_img = draw_json_polygons(img_name, json_file_name, class_1_img_folder_xpl, polygon_label_folder, image_size)
        pred_heatmap = grad_CAM.cam_pipeline_2_view(base_img_folders_class_1, img_name, image_size, model, last_conv_layer_name, cam_img_output_path=cam_img_output_path, segmented_img=segmented_img)
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

        olivine_iou = calc_iou_score(ground_truth_heatmap, pred_heatmap)
        olivine_iou_list.append(olivine_iou)

    # calculate IoU score for olivine images only
    olivine_mean_iou = calc_iou_score(ground_truth_added_heatmaps, predicted_binary_added_heatmaps)

    # predict heatmaps for non-olivine images in validation and test set and add to existing heatmaps
    for file in os.listdir(class_0_img_folder_xpl):
        # check if file is in val/test set
        for id in [*valset, *testset]:
            if id in file:
                pred_heatmap = grad_CAM.cam_pipeline_2_view(base_img_folders_class_0, file, image_size, model, last_conv_layer_name, draw_text=False)
                predicted_binary_heatmap = np.where(pred_heatmap > iou_threshold, 1, 0)
                ground_truth_heatmap = np.zeros(predicted_binary_heatmap.shape)
                predicted_binary_added_heatmaps = np.concatenate((predicted_binary_added_heatmaps, predicted_binary_heatmap.flatten()), axis=None)
                ground_truth_added_heatmaps = np.concatenate((ground_truth_added_heatmaps, ground_truth_heatmap.flatten()), axis=None)


    # calculating the iou value over all images combined 
    global_mean_iou = calc_iou_score(ground_truth_added_heatmaps, predicted_binary_added_heatmaps)

    max_olivine_iou = np.max(olivine_iou_list)
    min_olivine_iou = np.min(olivine_iou_list)

    return [global_mean_iou, olivine_mean_iou, max_olivine_iou, min_olivine_iou]



def calc_mean_segmentation_scores_multi_input(class_0_img_folder_ppl_0,
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
                                              cam_img_output_path, 
                                              xlsx_input_split_file):
    print("Calculating IoU score...")
    base_img_folders_class_0 = [class_0_img_folder_ppl_0, 
                                class_0_img_folder_ppl_30, 
                                class_0_img_folder_ppl_60,
                                class_0_img_folder_ppl_merged,
                                class_0_img_folder_xpl_0,
                                class_0_img_folder_xpl_30,
                                class_0_img_folder_xpl_60,
                                class_0_img_folder_xpl_merged,
                               ]
    base_img_folders_class_1 = [class_1_img_folder_ppl_0, 
                                class_1_img_folder_ppl_30, 
                                class_1_img_folder_ppl_60,
                                class_1_img_folder_ppl_merged,
                                class_1_img_folder_xpl_0,
                                class_1_img_folder_xpl_30,
                                class_1_img_folder_xpl_60,
                                class_1_img_folder_xpl_merged,
                               ]
    valset, testset = get_val_test_data(xlsx_input_split_file)
    predicted_binary_added_heatmaps = np.zeros([0])
    ground_truth_added_heatmaps = np.zeros([0])

    olivine_iou_list = []

    # predict heatmaps for olivine images
    for json_file_name in os.listdir(polygon_label_folder):
        img_name = get_json_img_name(json_file_name)
        image_id = "_".join(img_name.split('_')[:3])
        segmented_img = draw_json_polygons(img_name, json_file_name, class_1_img_folder_xpl_merged, polygon_label_folder, image_size)

        pred_heatmap = grad_CAM.cam_pipeline_multi_input(base_img_folders_class_1,
                                                         img_name,
                                                         image_size,
                                                         input_size,
                                                         model,
                                                         last_conv_layer_name,
                                                         cam_img_output_path=cam_img_output_path,
                                                         segmented_img=segmented_img)

        predicted_binary_heatmap = np.where(pred_heatmap > iou_threshold, 1, 0)
        heatmap_size = predicted_binary_heatmap.shape
        
        #add all predicted segmentations to one large vector
        predicted_binary_added_heatmaps = np.concatenate((predicted_binary_added_heatmaps, predicted_binary_heatmap.flatten()), axis=None)

        # search for corresponding labeled .npy file:
        for img_file in os.listdir(voc_label_folder):
            if image_id in img_file:
                ground_truth_heatmap = np.load(os.path.join(voc_label_folder, img_file))
                ground_truth_heatmap = resize(ground_truth_heatmap, heatmap_size)

                # add all ground truth segmentations to one large 1D vector
                ground_truth_added_heatmaps = np.concatenate((ground_truth_added_heatmaps, ground_truth_heatmap.flatten()), axis=None)

        olivine_iou = calc_iou_score(ground_truth_heatmap, pred_heatmap)
        olivine_iou_list.append(olivine_iou)

    # calculate IoU score for olivine images only
    olivine_mean_iou = calc_iou_score(ground_truth_added_heatmaps, predicted_binary_added_heatmaps)

    # predict heatmaps for non-olivine images in validation and test set and add to existing heatmaps
    for img_name in os.listdir(class_0_img_folder_xpl_merged):
        # check if file is in val/test set
        for id in [*valset, *testset]:
            if id in img_name:
                pred_heatmap = grad_CAM.cam_pipeline_multi_input(base_img_folders_class_0,
                                                                 img_name,
                                                                 image_size,
                                                                 input_size,
                                                                 model,
                                                                 last_conv_layer_name,
                                                                 draw_text=False)
                predicted_binary_heatmap = np.where(pred_heatmap > iou_threshold, 1, 0)
                ground_truth_heatmap = np.zeros(predicted_binary_heatmap.shape)
                predicted_binary_added_heatmaps = np.concatenate((predicted_binary_added_heatmaps, predicted_binary_heatmap.flatten()), axis=None)
                ground_truth_added_heatmaps = np.concatenate((ground_truth_added_heatmaps, ground_truth_heatmap.flatten()), axis=None)


    # calculating the iou value over all images combined 
    global_mean_iou = calc_iou_score(ground_truth_added_heatmaps, predicted_binary_added_heatmaps)

    max_olivine_iou = np.max(olivine_iou_list)
    min_olivine_iou = np.min(olivine_iou_list)

    return [global_mean_iou, olivine_mean_iou, max_olivine_iou, min_olivine_iou]