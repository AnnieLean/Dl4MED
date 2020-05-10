import torch
import torch.nn as nn
import torch.nn.functional as F
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

def ConvLayer(in_size, out_size, conv_stride = 2):
    conv_layer = nn.Sequential(
        nn.Conv2d(in_channels = in_size, out_channels = out_size,
                  kernel_size = 3, stride = conv_stride, padding = 1),
        nn.ReLU(),
        nn.BatchNorm2d(out_size),
        nn.AdaptiveMaxPool2d(out_size)
    )
    
    return conv_layer

class ResNetBlock(nn.Module):
    def __init__(self, in_size, temp_size, out_size):
        super(ResNetBlock, self).__init__()
        self.in_size = in_size
        self.temp_size = temp_size
        self.out_size = out_size
        self.resnet = nn.Sequential(
            nn.Conv2d(in_channels = in_size, out_channels = temp_size, kernel_size = 3,  padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = temp_size, out_channels = out_size, kernel_size = 3, padding = 1)
        )
        
    def forward(self, x):
        out = self.resnet(x)
        out = out + x
        out = F.relu(out)
        return out
        

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = ConvLayer(3, 16)
        self.resnet1 = ResNetBlock(16, 16, 16)
        self.conv2 = ConvLayer(16, 32)
        self.resnet2 = ResNetBlock(32, 32, 32)
        self.conv3 = ConvLayer(32, 64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1)
        self.bn = nn.BatchNorm2d(64) ## output size (64 * 32 * 32)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.classifier = nn.Linear(512, 5)
    
    def forward(self, x):
        out = self.resnet1(self.conv1(x))
        out = self.resnet2(self.conv2(out))
        out = self.conv3(out)
        out = self.relu(self.conv4(out))
        out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.classifier(out)
        
        return out

def train_model(model, model_name, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    acc_dict = {'train':[],'val':[]}
    loss_dict = {'train':[],'val':[]}
    best_acc = -1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 50)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs = data['img'].to(device)
                labels = data['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            
            # statistics
            acc_dict[phase].append(epoch_acc)
            loss_dict[phase].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if scheduler:
                scheduler.step()
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_dict,
                'accuracy': acc_dict,
            }, '{}_checkpoint.pt'.format(model_name))
            
        print()

    print('Best val acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_name + '_bt_acc.pt')
    
    json.dump(acc_dict, open(model_name + '_acc_dict.txt','w'))
    json.dump(loss_dict, open(model_name + '_loss_dict.txt','w'))
    
    # return model, acc_dict, loss_dict
    

if __name__ == '__main__':
    # Receive epoch numbers as argument inputs
    model_name = sys.argv[1]
    num_epochs = int(sys.argv[2])

    # Load dataloader
    train_loader = pkl.load(open("train_loader_512.pkl", "rb"))
    valid_loader = pkl.load(open("valid_loader_512.pkl", "rb"))
    dataloader = {'train': train_loader, 'val': valid_loader}

    # Detect if we have a GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build customized models
    cnn = ConvNet().to(device)
    optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    train_model(cnn, model_name, dataloader, criterion, optimizer, scheduler = None,
                num_epochs = num_epochs, is_inception=False)


    
        