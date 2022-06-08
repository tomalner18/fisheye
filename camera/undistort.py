from email.mime import base
from projection import Camera, RadialPolyCamProjection, CylindricalProjection, read_cam_from_json, \
    create_img_projection_maps
import pathlib
import os
from glob import glob

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as SciRot
# from matplotlib import pyplot as plt

base_dir = "/home/tom/Documents/dev/ha/od-repos/fisheye/datasets"
target_dir = os.path.join(base_dir, "cyl_woodscape")


def make_cylindrical_cam(cam: Camera):
    """generates a cylindrical camera with a centered horizon"""
    assert isinstance(cam.lens, RadialPolyCamProjection)
    # creates a cylindrical projection
    lens = CylindricalProjection(cam.lens.coefficients[0])
    rot_zxz = SciRot.from_matrix(cam.rotation).as_euler('zxz')
    # adjust all angles to multiples of 90 degree
    rot_zxz = np.round(rot_zxz / (np.pi / 2)) * (np.pi / 2)
    # center horizon
    rot_zxz[1] = np.pi / 2
    # noinspection PyArgumentList
    return Camera(
        rotation=SciRot.from_euler(angles=rot_zxz, seq='zxz').as_matrix(),
        translation=cam.translation,
        lens=lens,
        size=cam.size, principle_point=(cam.cx_offset, cam.cy_offset),
        aspect_ratio=cam.aspect_ratio
    )


def project_bboxes(img_name, img_path, base_cam, target_cam, split):
    ann_name = img_name + ".txt"
    ann_path = os.path.join(os.path.dirname(
        os.path.dirname(img_path)), "labels", ann_name)
    new_ann_path = os.path.join(target_dir, split, "labels", ann_name)
    with open(ann_path, 'r') as f:
        annot_data = f.readlines()

    # To store the new annotations
    annot_copy = []

    for i, ann in enumerate(annot_data):
        cls, cid, l, t, r, b = ann.split(",")

        # [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        base_points = np.array([(l, t), (r, t), (r, b), (l, b)], int)

        # Project the bounding boxes onto the target image
        world_point = base_cam.project_2d_to_3d(
            base_points, norm=np.ones(base_points.shape[0]))
        [new_tl, new_tr, new_br, new_bl] = target_cam.project_3d_to_2d(world_point)

        # We need to take the extremes to guarentee our new bbox fits
        new_l = int(min(new_tl[0], new_bl[0]))
        new_t = int(min(new_tl[1], new_tr[1]))
        new_r = int(max(new_tr[0], new_br[0]))
        new_b = int(max(new_bl[1], new_br[1]))

        new_ann = [cls, cid, str(new_l), str(new_t), str(new_r), str(new_b)]
        annot_copy.append(",".join(new_ann) + "\n")


    with open(new_ann_path, 'w') as t:
        t.writelines(annot_copy)

    print(new_ann_path)


def project(img_path):
    img_name = os.path.basename(img_path).split(".")[0]
    split = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
    cal_name = img_name + ".json"
    cal_path = os.path.join(base_dir, "calibration/calibration", cal_name)

    base_cam = read_cam_from_json(cal_path)
    target_cam = make_cylindrical_cam(base_cam)

    # project the image itself
    fisheye_image = cv2.imread(img_path)
    map1, map2 = create_img_projection_maps(base_cam, target_cam)
    target_image = cv2.remap(fisheye_image, map1, map2, cv2.INTER_CUBIC)

    # Save the projected image
    cv2.imwrite(os.path.join(target_dir, split,
                "images", os.path.basename(img_path)), target_image)

    # project the annotations
    project_bboxes(img_name, img_path, base_cam, target_cam, split)


def main():

    img_paths = sorted(
        glob(os.path.join(base_dir, "fish_woodscape", "train", "images", "*.png")))

    for img_path in img_paths:
        project(img_path)


if __name__ == "__main__":
    main()
