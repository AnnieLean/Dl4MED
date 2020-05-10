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

def train_model(model, model_name, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, scheduler=None):
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
    use_pretrained = sys.argv[2]
    num_epochs = int(sys.argv[3])

    # Load dataloader
    train_loader = pkl.load(open("train_loader_299.pkl", "rb"))
    valid_loader = pkl.load(open("valid_loader_299.pkl", "rb"))

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build inception models
    inception = models.inception_v3(pretrained=use_pretrained)
    num_ftrs = inception.fc.in_features
    inception.fc = nn.Linear(num_ftrs, 5)
    aux_num_ftrs = inception.AuxLogits.fc.in_features
    inception.AuxLogits.fc = nn.Linear(aux_num_ftrs, 5)
    inception.to(device)

    optimizer_ft = optim.SGD(inception.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    dataloader = {'train': train_loader, 'val': valid_loader}

    # Train and evaluate
    train_model(inception, model_name, dataloader, criterion, optimizer_ft, 
                num_epochs = num_epochs, is_inception=True)
