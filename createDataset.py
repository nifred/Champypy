import cv2
import argparse
import os
import pandas as pd

def image_list(path):
    img_path = []
    for dirs, subdirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                img_path.append(os.path.join(dirs, file))
    return img_path

def convert_image(pathname):
    classe = []
    list_image = []
    for img in pathname:
        image = cv2.imread(img, -1)
        image = cv2.resize(image,(28,28))
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = image.reshape(
                        image.shape[0]*image.shape[1]*image.shape[2]
                )
                list_image.append(image)
                if img.__contains__('\\'):
                    classe.append(img.split('\\')[-2])
                else:
                    classe.append(img.split('/')[-2])
    return list_image, classe


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="pictures directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(dir_path, 'dataset.csv'),
        help="pathname for save dataset"
    )
    args = parser.parse_args()
    if args.directory:
        pathname = image_list(args.directory)
    if pathname:
        list_image, classe = convert_image(pathname)
    df_image = pd.DataFrame(list_image)
    df_image['classe'] = classe
    if args.dataset:
        df_image.to_csv(args.dataset)
