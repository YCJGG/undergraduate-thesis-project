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
import networks
from networks import GANLoss, Generator, Discriminator
from torchvision import models
from BatchReader import DatasetProcessing



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type = int, default= 128, help = "batch size")
parser.add_argument('--g_input_size', type = int, default= 2048, help = "input size of generator")
parser.add_argument('--g_hidden_size', type = int, default= 4096, help = "hidden size of generator")
parser.add_argument('--g_output_size', type = int, default= 2048, help = "output size of generator")
parser.add_argument('--d_input_size', type = int, default= 4096, help = "input size of discriminator")
parser.add_argument('--d_hidden_size', type = int, default= 1024, help = "hidden size of discriminator")
parser.add_argument('--d_output_size', type = int, default= 64 , help = "output size of discriminator")
parser.add_argument('--h_input_size', type = int, default= 2048, help = "input size of Hashnet")
parser.add_argument('--h_hidden_size', type = int, default= 1024, help = "hidden size of Hashnet")
parser.add_argument('--bit', type = int, default= 64 , help = "output size of Hashnet")
parser.add_argument('--lrR', type = float, default = 2e-4, help = "learning rate of Resnet" )
parser.add_argument('--lrG', type = float, default = 2e-4, help = "learning rate of generator" )
parser.add_argument('--lrD', type = float, default = 2e-5, help = "learning rate of discriminator" )
parser.add_argument('--lrH', type = float, default = 2e-4, help = "learning rate of Hashnet" )
parser.add_argument('--beta1', type = float, default = 0.5, help = "beta1 for Adam optimizer" )
parser.add_argument('--beta2', type = float, default = 0.999, help = "beta2 for Adam optimizer" )
parser.add_argument('--train_epoch', type = int, default = 150, help = "training epochs")
parser.add_argument('--lamb', type = float, default = 10, help = "lambada")
opt = parser.parse_args()


# load labels
def LoadLabel(filename):
    fp = open(filename,'r')
    labels = [x.strip().split()[1] for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int,labels)))

def EncodingOnehot(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot

#DATA_DIR = './data/256x256'
DATABASE_FILE = 'all.list'
TRAIN_FILE = 'train.list'
TEST_FILE = 'test.list'
nclasses = 101

transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
dset_test = DatasetProcessing(TEST_FILE, transformations)
dset_train = DatasetProcessing(TRAIN_FILE, transformations)

num_train, num_test = len(dset_train) , len(dset_test)

train_loader = DataLoader(dset_train,batch_size=opt.batch_size,shuffle=True,num_workers=4)
test_loader = DataLoader(dset_test,batch_size=opt.batch_size,shuffle=False,num_workers=4)


train_labels = LoadLabel(TRAIN_FILE)
train_labels_onehot = EncodingOnehot(train_labels, nclasses)
test_labels = LoadLabel(TEST_FILE)
test_labels_onehot = EncodingOnehot(test_labels, nclasses)
Y = train_labels_onehot

# network

R = models.resnet50(pretrained = True)
R.fc = nn.Linear(2048,2048)
#G = networks.Generator(opt.g_input_size,opt.g_hidden_size,opt.g_output_size)
G = R
D = networks.Discriminator(opt.d_input_size,opt.d_hidden_size,opt.d_output_size)
H = networks.Hashnet(opt.h_input_size,opt.h_hidden_size,opt.bit)

#R = nn.DataParallel(R)
G = nn.DataParallel(G)
D = nn.DataParallel(D)
H = nn.DataParallel(H)

#print(R)

#R.cuda()
G.cuda()
D.cuda()
H.cuda()

#loss 
criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionGAN = criterionGAN.cuda()
criterionL1 = criterionL1.cuda()

# Adam optimizer
#R_optimizer = optim.Adam(R.parameters(), lr = opt.lrR, betas = (opt.beta1,opt.beta2))
G_optimizer = optim.Adam(G.parameters(), lr = opt.lrG, betas = (opt.beta1,opt.beta2))
D_optimizer = optim.Adam(D.parameters(), lr = opt.lrD, betas = (opt.beta1,opt.beta2))
H_optimizer = optim.Adam(H.parameters(), lr = opt.lrH, betas = (opt.beta1,opt.beta2))

# training

print("###training start~~~~")

# initialize the B and H
B = torch.sign(torch.randn(num_train, opt.bit))
H_ = torch.zeros(num_train,opt.bit)

max_map = 0
itr = 0
file = open(str(opt.lrG)+'_' + str(opt.lrD)+'_' + str(opt.lrH)+'_' + str(opt.bit) + '.log','a')
for epoch in range(opt.train_epoch):
    # adjust the lr 
    H_optimizer.param_groups[0]['lr'] = opt.lrH*(0.1**(epoch//100))

    # E step
    temp1 = Y.t().mm(Y) +torch.eye(nclasses)
    temp1 = temp1.inverse()
    temp1 = temp1.mm(Y.t())
    E = temp1.mm(B)
    #print(D)
    # B step 
    B = torch.sign(Y.mm(E) + 1e-5 * H_)

    R.train()
    G.train()
    D.train()
    H.train()
    #F step
    for iteration, batch in enumerate(train_loader, 0):
        seg1 = batch[0]
        seg2 = batch[1]
        seg3 = batch[2]
        partial = batch[3]
        label = batch[4]
        batch_ind = batch[5]

        seg1, seg2,seg3,partial = Variable(seg1.cuda()), Variable(seg2.cuda()),Variable(seg3.cuda()),Variable(partial.cuda())
        
        #generate the full feature
        ff1 = G(seg1)
        ff2 = G(seg2)
        ff3 = G(seg3)
        ff = (ff1 + ff2 + ff3)/3
        pf = G(partial)
        # generate partial feature to full feature

        fakef  = G(pf)
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        # train generator D
        D_optimizer.zero_grad()
        # train with fake
        fake_ab = torch.cat((pf,fakef),2)
        # bs * 2 * 4096 
        pred_fake = D.forward(fake_ab.detach())
        # bs * 2 * 64
        loss_d_fake = criterionGAN(pred_fake, False)
        # train with real
        real_ab = torch.cat((pf,ff),2)
        # bs * 2 * 4096
        pred_real = D.forward(real_ab)
        # bs * 2 * 64
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()

        D_optimizer.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################

        G_optimizer.zero_grad()
         # First, G(A) should fake the discriminator
        fake_ab = torch.cat((pf,fakef),2)
        
        pred_fake = D.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)
         # Second, G(A) = B
        loss_g_l1 = criterionL1(fakef, ff) * opt.lamb
        
        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward()
        G_optimizer.step()

        H_optimizer.zero_grad()
        fakef = G(pf)
        H_fake = H(fakef)
        H_real = H(ff)
        temp = torch.zeros(H_real.data.size())
        for i , ind in enumerate(batch_ind):
            temp[i, :] = B[ind, :]
            H_[ind, :] = H_real.data[i]
        temp = Variable(temp.cuda())
        regterm1 = (temp - H_fake).pow(2).sum()
        regterm2 = (temp - H_real).pow(2).sum()
        regterm3 = (H_real - H_fake).pow(2).sum()
        
        H_loss = (regterm1 +regterm2 + regterm3)/opt.batch_size
        
        H_loss.backward()
        H_optimizer.step()

        
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} Loss_H: {:.4f}".format(
            epoch, itr, len(train_loader)*opt.batch_size, loss_d.data[0], loss_g.data[0],H_loss.data[0]))
        itr+=1
