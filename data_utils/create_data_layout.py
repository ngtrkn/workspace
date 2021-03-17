from glob import glob
import os

def create_train_data(data_list: list):
    x = []; y = []
    for data_path in data_list:
        x_ = glob(data_path + "/**/*", recursive=True)
        x_ = [item for item in x_ if ("GT" not in os.path.basename(item)) and ('mask' not in os.path.basename(item))]

        x_data = []
        y_data = []
        for item in x_:
            y_item = ".".join(item.split(".")[:-1])
            y_GT0 = y_item + "_GT0.jpg"
            y_GT1 = y_item + "_GT1.jpg"
            y_GT2 = y_item + "_GT2.jpg"
            y_GT3 = y_item + "_mask.png"
            #TODO: check if not empty item
            if any([os.path.isfile(gt) for gt in [y_GT0, y_GT1, y_GT2, y_GT3]]):
                x_data.append(item)
                y_data.append([gt if os.path.isfile(gt) else None for gt in [y_GT0, y_GT1, y_GT2, y_GT3]])

        x.extend(x_data)
        y.extend(y_data)
    
    return x, y