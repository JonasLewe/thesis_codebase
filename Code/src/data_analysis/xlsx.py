# This file parses all information from the folder of a single model run into an xlsx file
from openpyxl import load_workbook


def parse_model_output_to_xlsx(model_name, model_type, epochs, learning_rate, batch_size, seed_value, mean_iou_score, gpu, xlsx_file):
    wb = load_workbook(filename=xlsx_file)
    page = wb.active
    data = [model_name, model_type, epochs, learning_rate, batch_size, seed_value, mean_iou_score, gpu] 
    page.append(data)
    wb.save(filename=xlsx_file)
