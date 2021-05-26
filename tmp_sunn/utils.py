
import os
from glob import glob
from sklearn.model_selection import train_test_split


def get_images(data_dir, class_list, for_split=True):
    img_list = []
    extension = ['*.jpg', '*.png', '*.gif', '*.jpeg', '*.bmp']

    if for_split:
        train_list, val_list = [], []

        for cls_name in class_list:
            tmp_list = []
            for ext in extension:
                tmp_list += glob(os.path.join(data_dir, cls_name, ext))
            train_img_list, val_img_list = train_test_split(tmp_list[:4000], test_size=0.1)
            train_list += train_img_list
            val_list += val_img_list

        return train_list, val_list

    for cls_name in class_list:
        for ext in extension:
            img_list += glob(os.path.join(data_dir, cls_name, ext))

    return img_list

# todo : logger 설정
def set_logger(logger_dir):
    return 0
