import torch
from BatchReader import DatasetProcessing
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models
import argparse
import networks
import torch.optim as optim


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default= 128, help = "batch size")
parser.add_argument('--g_input_size', type = int, default= 4096, help = "input size of generator")
parser.add_argument('--g_hidden_size', type = int, default= 8192, help = "hidden size of generator")
parser.add_argument('--g_output_size', type = int, default= 4096, help = "output size of generator")
parser.add_argument('--d_input_size', type = int, default= 4096, help = "input size of discriminator")
parser.add_argument('--d_hidden_size', type = int, default= 1024, help = "hidden size of discriminator")
parser.add_argument('--d_output_size', type = int, default= 64 , help = "output size of discriminator")
parser.add_argument('--h_input_size', type = int, default= 4096, help = "input size of Hashnet")
parser.add_argument('--h_hidden_size', type = int, default= 1024, help = "hidden size of Hashnet")
parser.add_argument('--bit', type = int, default= 32 , help = "output size of Hashnet")
parser.add_argument('--lrG', type = float, default = 1e-4, help = "learning rate of generator" )
parser.add_argument('--lrD', type = float, default = 1e-4, help = "learning rate of discriminator" )
parser.add_argument('--lrH', type = float, default = 1e-4, help = "learning rate of Hashnet" )
parser.add_argument('--beta1', type = float, default = 0.5, help = "beta1 for Adam optimizer" )
parser.add_argument('--beta2', type = float, default = 0.999, help = "beta2 for Adam optimizer" )
parser.add_argument('--train_epoch', type = int, default = 100, help = "training epochs")
opt = parser.parse_args()

# load labels
def LoadLabel(filename):
    fp = open(filename,'r')
    labels = [x.strip().split()[2] for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int,labels)))

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


#dataloader
TRAIN_DIR = 'train.list'
TEST_DIR = 'test.list'
nclasses = 101


train_data = DatasetProcessing(TRAIN_DIR)
test_data = DatasetProcessing(TEST_DIR)

num_train, num_test = len(train_data) , len(test_data)

train_loader = DataLoader(train_data,batch_size = opt.batch_size, shuffle = True, num_workers = 4)
test_loader = DataLoader(test_data,batch_size = opt.batch_size, shuffle = False, num_workers = 4)


train_labels = LoadLabel(TRAIN_DIR)
train_labels_onehot = EncodingOnehot(train_labels, nclasses)
print(train_labels_onehot)


G = networks.Generator(opt.g_input_size,opt.g_hidden_size,opt.g_output_size)
D = networks.Discriminator(opt.d_input_size,opt.d_hidden_size,opt.d_output_size)
H = networks.Hashnet(opt.h_input_size,opt.h_hidden_size,opt.bit)
# print(G)
# print(D)
# print(H)
G.cuda()
D.cuda()
H.cuda()


#loss 
BCE_loss = nn.BCELoss().cuda()
MSE_loss = nn.MSELoss().cuda()
L1_loss = nn.L1Loss().cuda()



# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr = opt.lrG, betas = (opt.beta1,opt.beta2))
D_optimizer = optim.Adam(D.parameters(), lr = opt.lrD, betas = (opt.beta1,opt.beta2))
H_optimizer = optim.Adam(H.parameters(), lr = opt.lrH, betas = (opt.beta1,opt.beta2))

# training

print("###training start~~~~")




#for epoch in range(opt.train_epoch):







