import torch
from torchvision import transforms, datasets
import json
import copy
import os
import argparse


def save_wights(model, optimizer, args, classifier):
    wights = {'arch': args.arch,
                  'model': model,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'classifier': classifier,
                  'epochs': args.epochs,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(wights, 'checkpoint.pth')


def load_wights(filepath):
    wights = torch.load(filepath)
    wights_model = wights['model']
    wights_model.classifier = wights['classifier']
    learning_rate = wights['learning_rate']
    epochs = wights['epochs']
    optimizer = wights['optimizer']
    wights_model.load_state_dict(wights['state_dict'])
    wights_model.class_to_idx = wights['class_to_idx']

    return wights_model


def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names