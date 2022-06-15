# This file parses all information from the folder of a single model run into an xlsx file
from openpyxl import load_workbook


def parse_model_output_to_xlsx(model_name,
                               model_type,
                               input_size,
                               epochs,
                               learning_rate,
                               batch_size,
                               iou_threshold,
                               dropout_value,
                               regularization,
                               seed_value,
                               num_of_runs,
                               mean_iou_score,
                               mean_iou_olivine,
                               max_iou_olivine,
                               min_iou_olivine,
                               precision,
                               recall,
                               fscore,
                               gpu,
                               comments,
                               xlsx_file):
    wb = load_workbook(filename=xlsx_file)
    page = wb.active
    data = [model_name,
            model_type,
            input_size,
            epochs,
            learning_rate,
            batch_size,
            iou_threshold,
            dropout_value,
            regularization,
            seed_value,
            num_of_runs,
            mean_iou_score,
            mean_iou_olivine,
            max_iou_olivine,
            min_iou_olivine,
            precision,
            recall,
            fscore,
            gpu,
            comments,
           ] 
    page.append(data)
    wb.save(filename=xlsx_file)
