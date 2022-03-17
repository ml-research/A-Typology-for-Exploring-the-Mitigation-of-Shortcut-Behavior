"""Example calls to plot heatmaps and calculate WR on ISIC19."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils import data

from learner.models import dnns
from learner.learner import Learner
from data_store.datasets import decoy_mnist, decoy_mnist_CE_augmented
from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, RBRLoss
import util
import explainer
import matplotlib.pyplot as plt
import argparse
import os


# +
# __import__("pdb").set_trace()
parser = argparse.ArgumentParser(description='XIL EVAL')
parser.add_argument('-m', '--mode', default='RRR', type=str, choices=['Vanilla','RRR','RRR-G','HINT','CDEP','CE','RBR'],
                    help='Which XIL method to test?')
parser.add_argument('--dataset', default='Mnist', type=str, choices=['Mnist','FMnist'],
                    help='Which dataset to use?')
parser.add_argument('--run', default=0, type=int,
                    help='Which Seed?')

args = parser.parse_args()
# -

# Get cpu or gpu device for training.
DEVICE = "cuda"
SEED = [1, 10, 100, 1000, 10000]
SHUFFLE = True
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 50
SAVE_BEST = True
VERBOSE_AFTER_N_EPOCHS = 2

print("\nUsing {} device".format(DEVICE))

# +
############# Initalize dataset and dataloader
if args.dataset == 'Mnist':
    train_dataloader, test_dataloader = decoy_mnist(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
    if args.mode == 'Vanilla':
        args.reg = None
        args.mode = 'CEL'
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'RRR':
        args.reg = 10
        loss_fn = RRRLoss(args.reg)
    elif args.mode == 'RBR':
        args.reg = 100000
        loss_fn = RBRLoss(args.reg)
    elif args.mode == 'RRR-G':
        args.reg = 1
        loss_fn = RRRGradCamLoss(args.reg)
        args.mode = 'RRRGradCAM'
    elif args.mode == 'HINT':
        train_dataloader, val_dataloader = decoy_mnist(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, \
                                       hint_expl=True)
        args.reg = 100
        loss_fn = HINTLoss(args.reg, last_conv_specified=True, upsample=True)
    elif args.mode == 'CE':
        train_dataloader, val_dataloader = decoy_mnist_CE_augmented(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'CDEP':
        args.reg = 1000000  
        loss_fn = CDEPLoss(args.reg)
        
elif args.dataset == 'FMnist':
    train_dataloader, test_dataloader = decoy_mnist(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
    if args.mode == 'Vanilla':
        args.reg = None
        args.mode == 'CEL'
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'RRR':
        args.reg = 10
        loss_fn = RRRLoss(args.reg)
    elif args.mode == 'RBR':
        args.reg = 1000000
        loss_fn = RBRLoss(args.reg)
    elif args.mode == 'RRR-G':
        args.reg = 10
        loss_fn = RRRGradCamLoss(args.reg)
        args.mode = 'RRRGradCAM'
    elif args.mode == 'HINT':
        train_dataloader, val_dataloader = decoy_mnist(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, \
                                               hint_expl=True)
        args.reg = 0.00001
        loss_fn = HINTLoss(args.reg, last_conv_specified=True, upsample=True)
    elif args.mode == 'CE':
        train_dataloader, val_dataloader = decoy_mnist_CE_augmented(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'CDEP':
        args.reg = 2000000  
        loss_fn = CDEPLoss(args.reg)
# -


i = args.run
util.seed_all(SEED[i])
model = dnns.SimpleConvNet().to(DEVICE)
MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.reg}--seed={SEED[i]}--run={i}'
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
learner = Learner(model, loss_fn, optimizer, DEVICE, MODELNAME)
learner.fit(train_dataloader, test_dataloader, EPOCHS, save_best=SAVE_BEST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)
avg0.append(learner.score(test_dataloader, nn.CrossEntropyLoss())[0])
