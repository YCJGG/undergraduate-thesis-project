import torch
from BatchReader import DatasetProcessing
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models
import argparse
import networks
from networks import GANLoss
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
parser.add_argument('--lamb', type = float, default = 10, help = "lambada")
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
test_labels = LoadLabel(TEST_DIR)
test_labels_onehot = EncodingOnehot(test_labels, nclasses)
Y = train_labels_onehot


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
criterionGAN = GANLoss()
criterionL1 = nn.L1Loss()
criterionGAN = criterionGAN.cuda()
criterionL1 = criterionL1.cuda()



# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr = opt.lrG, betas = (opt.beta1,opt.beta2))
D_optimizer = optim.Adam(D.parameters(), lr = opt.lrD, betas = (opt.beta1,opt.beta2))
H_optimizer = optim.Adam(H.parameters(), lr = opt.lrH, betas = (opt.beta1,opt.beta2))

# training

print("###training start~~~~")

# initialize the B and H
B = torch.sign(torch.randn(num_train, opt.bit))
H_ = torch.zeros(num_train,opt.bit)


for epoch in range(opt.train_epoch):
    # E step
    temp1 = Y.t().mm(Y) +torch.eye(nclasses)
    temp1 = temp1.inverse()
    temp1 = temp1.mm(Y.t())
    E = temp1.mm(B)
    #print(D)
    # B step 
    B = torch.sign(Y.mm(E) + 1e-5 * H_)

    G.train()
    D.train()
    H.train()
    #F step
    for iteration, batch in enumerate(train_loader, 0):
        ff = batch[0]
        pf = batch[1]
        label = batch[2]
        batch_ind = batch[3]

        ff, pf = Variable(ff.cuda()), Variable(pf.cuda())
        

        # generate partial feature to full feature

        fakef  = G(pf)
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################
        # train generator D
        D_optimizer.zero_grad()
        # train with fake
        fake_ab = torch.cat((pf,fakef),1)
        # bs * 2 * 4096 
        pred_fake = D(fake_ab)
        # bs * 2 * 64
        loss_d_fake = criterionGAN(pred_fake, False)
        # train with real
        real_ab = torch.cat((pf,ff),1)
        # bs * 2 * 4096
        pred_real = D(real_ab)
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
        fake_ab = torch.cat((pf,fakef),1)
        
        pred_fake = D(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)
         # Second, G(A) = B
        loss_g_l1 = criterionL1(fakef, ff) * opt.lamb
        
        loss_g = loss_g_gan + loss_g_l1
        
        loss_g.backward()
        G_optimizer.step()
        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(train_loader), loss_d.data[0], loss_g.data[0]))








