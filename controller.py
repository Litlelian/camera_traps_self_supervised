import cv2
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap
import tkinter as tk
from tkinter import filedialog

import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import resnet18
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from demo_ui import Ui_Form

class SimCLRModel(nn.Module):
    def __init__(self):
        super(SimCLRModel, self).__init__()

        self.enc = resnet18() 
        self.feature_dim = self.enc.fc.in_features
        self.projection_dim = 128 
        self.proj_hidden = 512
        
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.proj_hidden),
            nn.ReLU(),
            nn.Linear(self.proj_hidden, self.projection_dim)
        )
        self.enc.fc = nn.Identity()   

    def forward(self, x):
        x = self.enc(x)
        x = self.projector(x)
        return x

class Dialog_controller(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setup_control()
        self.graph_path = ''
        self.graph = None
        self.label = pd.read_csv('label.csv')

        self.transform = transforms.Compose([
                transforms.Resize([112,112]), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
            ])
        self.model = SimCLRModel()
        self.checkpoint = torch.load('./cct20_resnet18_simclr_2022_12_10__11_32_48.pt',map_location='cpu')
        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.eval()
        self.pca = joblib.load('pca.m')
        self.KM = joblib.load('km.pkl')
    
    def setup_control(self):
        # Load image
        self.ui.load_image.clicked.connect(self.load)
        # Inference
        self.ui.inf.clicked.connect(self.inference)


    def load(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        if len(file_path) != 0:
            self.graph_path = file_path
            self.graph = Image.open(file_path)
            self.ui.img.setPixmap(QPixmap(self.graph_path))
            self.ui.img.setScaledContents(True)
            
    def inference(self):
        if self.graph is None:
            print('Error!! Plaese load image first')
        else:
            pred = self.model(self.transform(self.graph).unsqueeze(0))
            preds_pca = self.pca.transform(pred.detach())
            ans = self.KM.predict(preds_pca)

            self.ui.label_2.setText('prediction1: {}, \nprediction2: {}'.format(
                self.label.loc[int(ans[0].item()), 'pred1'],
                self.label.loc[int(ans[0].item()), 'pred2'],
                ))
