from glob import glob
import os

def create_train_data(data_list: list):
    x = []; y = []
    for data_path in data_list:
        x_ = glob(data_path + "/images/training/*.jpg")

        x_data = []
        y_data = []
        for item in x_:
            y_item = item.replace("images", "annotations").replace(".jpg", ".png")
            #TODO: check if not empty item
            if os.path.isfile(y_item):
                x_data.append(item)
                y_data.append([y_item])

        x.extend(x_data)
        y.extend(y_data)
    
    return x, y