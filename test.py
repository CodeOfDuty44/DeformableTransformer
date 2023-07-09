import argparse
from config import get_config
# from models.dat import build_model
import torch
from models.dat import DAT
import torch.optim as optim 
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import warnings
warnings.filterwarnings("ignore")
def main():
    model_path = "weight/DAT_classifier_imagenet.tar"
    img_dir = "../data/tiny_test/"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = DAT(224, 1000).to(device="cpu")
    model.train()


    pretrained = torch.load(model_path)
    model.load_state_dict(pretrained["model"])
    model.eval()
    imgs = os.listdir(img_dir)
    tr_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    cls_id_to_str_path = "id2str.txt"
    with open(cls_id_to_str_path) as f:
        cls_id_to_str = f.readlines()

    for img_path in imgs:
        while True:
            img_path = os.path.join(img_dir, img_path )
            img = cv2.imread(img_path)
            img = tr_trans(img).to("cpu").unsqueeze(0)
            out = model(img)
            out = torch.argmax(out)
            cls_str = cls_id_to_str[out].split("'")[1]
        print(img_path, " : " , cls_str )


if __name__ == "__main__":
    main()

