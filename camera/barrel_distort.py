from wand.image import Image
import numpy as np
import cv2
import os
from glob import glob
import argparse
from tqdm import tqdm
import os
from pathlib import Path
import math

base_dir = "/home/tom/Documents/dev/ha/od-repos/OD-Labelling-Tool/datasets/dvc-dataset/object-detection/bdd100k"

fv_params = {"k1": 339.749,
             "k2": -31.988,
             "k3": 48.275,
             "k4": -7.201, }

# chi = sqrt( X ** 2 + Y ** 2)
# theta = arctan2( chi, Z ) = pi / 2 - arctan2( Z, chi )
# rho = rho(theta)
# u’ = rho * X / chi if chi != 0 else 0
# v’ = rho * Y / chi if chi != 0 else 0
# u = u’ + cx + width / 2 - 0.5
# v = v’ * aspect_ratio + cy + height / 2 - 0.5

# rho(theta) = k1 * theta + k2 * theta ** 2 + k3 * theta ** 3 + k4 * theta ** 4,


def rho(theta):
    return fv_params["k1"] * theta + fv_params["k2"] * (theta ** 2) + fv_params["k3"] * (theta ** 3) + fv_params["k4"] * (theta ** 4)


def distort_point(img, x, y):
    
    chi = math.sqrt(x ** 2 + y ** 2)
    # theta = np.arctan2(chi, Z) = pi / 2 - arctan2(Z, chi)
    # rho = rho(theta)
    # u_p = rho * X / chi if chi != 0 else 0
    # v_p = rho * Y / chi if chi != 0 else 0
    # u = u_p + cx + width / 2 - 0.5
    # v = v_p * aspect_ratio + cy + height / 2 - 0.5

def distort(img_path):
    img = cv2.imread(img_path)
    rows, cols, _ = img.shape
    for i in range(rows):
        for j in range(cols):
            distort_point(img, i - (rows / 2), j - (cols / 2))



img_paths = sorted(glob(os.path.join(base_dir, "images/10k/test", "*.jpg")))
annotations_paths = sorted(glob(os.path.join(
    base_dir, "labels/10k/test", "*.txt")))


for idx, img_path in enumerate(img_paths):
    file_name = os.path.basename(img_path).split(".")[0]
    img_name = file_name + ".jpg"
    ann_name = file_name + ".txt"
    ann_path = os.path.join(base_dir, "images/10k/test", ann_name)

    # img = cv2.imread(img_path)
    # distort(img_path)

    with Image(filename=img_path) as img:
        print(img.size)
        print(img_path)
        img.virtual_pixel = 'transparent'
        # distort(img_path)
        img.distort('barrel', (0.25, 0.1, 0.0))
        # img.distort('barrel', (fv_params["k1"], fv_params["k2"], fv_params["k3"], fv_params["k4"]))
        img.save(filename='checks_barrel.png')
        # convert to opencv/numpy array format
        img_opencv = np.array(img)

    # display result with opencv
    cv2.imshow("BARREL", img_opencv)
    cv2.waitKey(0)

    # os.rename(img_path, os.path.join(base_dir, split, "images", img_name))
    # os.rename(ann_path, os.path.join(base_dir, split, "labels", ann_name))
