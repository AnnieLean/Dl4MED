from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import pickle as pkl
from cnn_train import *
# from inception_train import *
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, RandomSampler, SequentialSampler
import torchvision.transforms as transforms
from PIL import Image
import pickle as pkl
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
from data_preprocess import RetinopathyDataset
import pandas as pd
import numpy as np
import json
import copy
import sys
import os

def pred_label(model, dataloader, is_inception = False):
    model.to(device)
    model.eval()
    
    output_probs = []
    label_list = []
    
    for data in dataloader:
        inputs = data['img'].to(device)
        label = data['label'].to(device)

        if is_inception:
            outputs, _ = model(inputs)
            output = F.softmax(outputs,dim=1)
        else:
            output = F.softmax(model(inputs),dim=1)

        output_probs.append(output.cpu().data.numpy())
        label_list.extend(label.cpu().data.numpy())
    
    output_probs = np.vstack(output_probs)
    
    y_score = output_probs
    y_label = label_list
    y_pred = np.argmax(y_score, axis=1)
    y_bi_target = label_binarize(label_list, classes=[0, 1, 2, 3, 4])

    return y_score, y_pred, y_label, y_bi_target


def plot_auc(y_score, y_target):
    n_classes = 5

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_target[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.title('AUC of Three Classes')
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label = 'AUC = {:.4f} of Class {}'.format(roc_auc[i], i))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig('auc.png')

if __name__ == '__main__':
    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load the correct testloader
    test_loader = pkl.load(open("test_loader_512.pkl", "rb"))
    # test_loader = pkl.load(open("test_loader_299.pkl", "rb"))

    # uncomment the model in need
    # inception = models.inception_v3(pretrained=use_pretrained)
    # num_ftrs = inception.fc.in_features
    # inception.fc = nn.Linear(num_ftrs, 5)
    # aux_num_ftrs = inception.AuxLogits.fc.in_features
    # inception.AuxLogits.fc = nn.Linear(aux_num_ftrs, 5)
    # inception.load_state_dict(torch.load('inceptionV3_bt_acc.pt'))

    cnn = ConvNet().to(device)
    cnn.load_state_dict(torch.load('ConvNet_bt_acc.pt'))

    # score, pred, true, target = pred_label(inception, test_loader)
    score, pred, true, target = pred_label(cnn, test_loader)

    # auc_plot
    plot_auc(score, target)
