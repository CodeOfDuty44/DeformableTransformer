import os
import xml.etree.ElementTree as ET
import numpy as np
import shutil

ld = os.listdir("/home/ubuntu/Imagenet/imagenet/ILSVRC/Data/CLS-LOC/train")
im_file_path = "/home/ubuntu/Imagenet/imagenet/ILSVRC/Data/CLS-LOC/val"
ann_file_path = "/home/ubuntu/Imagenet/imagenet/ILSVRC/Annotations/CLS-LOC/val"

os.mkdir("val_foldered")
for i in ld:
    os.mkdir(os.path.join("val_foldered",i))
im_list = np.sort(np.array(os.listdir(im_file_path)))
ann_list = np.sort(np.array(os.listdir(ann_file_path)))
for i in range(len(im_list)):
    im_dir = os.path.join(im_file_path,im_list[i])
    ann_dir = os.path.join(ann_file_path,ann_list[i])
    ann_lbl = ET.parse(ann_dir).getroot()[5][0].text
    # print(ann_lbl)
    shutil.copyfile(im_dir, os.path.join(os.getcwd(), "val_foldered",ann_lbl, im_list[i]))

    