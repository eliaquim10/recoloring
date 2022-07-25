from options.train_options import TrainOptions
import cv2
import subprocess

from os import *
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
    opt = TrainOptions().parse()
    IMAGE_DIR = opt.train_img_dir
    IMAGE_ID_LIST = [f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]
    dataset_size = len(IMAGE_ID_LIST)
    print("dataset_size", dataset_size)
    quantidade = 0
    for epoch, image_id in zip(range(dataset_size), IMAGE_ID_LIST):
        try:
            image = cv2.imread("{}/{}".format(IMAGE_DIR, image_id))
            if image.shape:
                pass
            # print(image.shape, epoch)
            quantidade+=1
        except:
            subprocess.run(["rm", "{}/{}".format(IMAGE_DIR, image_id)], capture_output=True)
            
    print("quantidade", quantidade)
            
# python visao.py --train_img_dir /opt/notebooks/train_data/dataset_updated/training_set/painting
# python visao.py --train_img_dir /opt/notebooks/test_data/painting