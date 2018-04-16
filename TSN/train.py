import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from os import listdir
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os, time, pickle, argparse
import cv2
import torch.nn.functional as F
from torchvision import models
from BatchReader import DatasetProcessing


#DATA_DIR = './data/256x256'
DATABASE_FILE = 'all.list'
TRAIN_FILE = 'train.list'
TEST_FILE = 'test.list'
nclasses = 101
