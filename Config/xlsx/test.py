from openpyxl import load_workbook

wb = load_workbook("train_val_test_split.xlsx")
ws = wb["Test"]
for row in ws:
    file_id = "_".join([row[1].value,row[2].value])
    print(file_id)
    

