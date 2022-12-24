import cv2
import os
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import joblib

import torchvision
import torchvision.transforms as transforms

import pickle
from tqdm import tqdm

from controller import SimCLRModel

transform = transforms.Compose([
                transforms.Resize([112,112]), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
            ])

if __name__ == '__main__':
    model_file = "./cct20_resnet18_simclr_2022_12_10__11_32_48.pt"
    image_dir = "cis_test_images"

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SimCLRModel()
    checkpoint = torch.load(model_file,map_location=dev)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    pca = joblib.load('pca.m')
    KM = joblib.load('km.pkl')

    preds_clr = []
    allfile = []
    for filename in tqdm(os.listdir(image_dir)):
        if filename[-3:] == 'jpg':
            image=Image.open(os.path.join(image_dir,filename))
            img = transform(image).unsqueeze(0)
            output = model(img).tolist()
            preds_clr.append(output)
            allfile.append(filename)
    preds_clr = torch.Tensor(preds_clr).squeeze(1)
    pca.fit(preds_clr)
    preds_clr = pca.transform(preds_clr)
    KM.fit(preds_clr)
    df = pd.DataFrame(columns=['image', 'cluster', 'gt_class'])
    for i,name in enumerate(allfile):
        gt_class = os.path.splitext(name)[0].split('_')[-1]
        df = pd.concat(
            [
                pd.DataFrame([[name, KM.labels_[i], gt_class]], columns=df.columns), df
            ], ignore_index=True
            )

    df.to_csv('class_label.csv')
    joblib.dump(pca, 'pca.m')
    joblib.dump(KM, 'km.pkl')
