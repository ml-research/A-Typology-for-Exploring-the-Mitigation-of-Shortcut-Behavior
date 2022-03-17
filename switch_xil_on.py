# +
import logging

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from learner.models import dnns
from learner.learner import Learner
from data_store.datasets import decoy_mnist, decoy_mnist_CE_augmented, isic_2019_hint, isic_2019
from xil_methods.xil_loss import RRRLoss, RRRGradCamLoss, CDEPLoss, HINTLoss, RBRLoss
import util
import explainer
import argparse

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 256
LR = 0.001
SAVE_LAST = True
VERBOSE_AFTER_N_EPOCHS = 2
DISABLE_FIRST_EPOCHS = 50
SEED = [1, 10, 100, 1000, 10000]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Using {DEVICE} device]")

parser = argparse.ArgumentParser(description='XIL EVAL')
parser.add_argument('-m', '--mode', default='RRR', type=str, choices=['RRR','RRR-G','HINT','CDEP','CE','RBR'],
                    help='Which XIL method to test?')
parser.add_argument('--data', default='MNIST', type=str, choices=['MNIST', 'FMNIST'],
                    help='Which explainer to use?')
parser.add_argument('--reg', default=1000, type=float, 
                    help='Which explainer to use?')

args = parser.parse_args()
methods = [args.mode]

# +
if args.data == 'MNIST':
    for n in range(5):
        util.seed_all(SEED[n])
        for meth in methods:
            if meth == 'RRR':
                train_dataloader, test_dataloader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE)      
                reg = 100
                loss_fn = RRRLoss(reg)
            elif meth == 'RBR':
                train_dataloader, test_dataloader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE)      
                reg = 1000000
                loss_fn = RBRLoss(args.reg)
            elif meth == 'RRR-G':
                train_dataloader, test_dataloader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE)      
                reg = 1
                loss_fn = RRRGradCamLoss(reg)
            elif meth == 'HINT':
                train_dataloader, test_dataloader = decoy_mnist(hint_expl=True, device=DEVICE, batch_size=BATCH_SIZE)
                reg = 1000
                loss_fn = HINTLoss(args.reg, last_conv_specified=True, upsample=True)
            elif meth == 'CE':
                train_dataloader, test_dataloader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE)      
                reg = None
                loss_fn = nn.CrossEntropyLoss()
            elif meth == 'CDEP':
                train_dataloader, test_dataloader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE)      
                reg = 1000000  
                loss_fn = CDEPLoss(args.reg)
            print(f'{meth}-{n}')
            model = dnns.SimpleConvNet().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            modelname = f'SwitchOn--DecoyMNIST-CNN-{meth}--seed={SEED[n]}--reg={args.reg}'
            learner = Learner(model, loss_fn, optimizer, DEVICE, modelname)
            if meth == 'CE':
                learner.fit(train_dataloader, test_dataloader, DISABLE_FIRST_EPOCHS, save_last=SAVE_LAST)
                train_loader, test_loader = decoy_mnist_CE_augmented(device=DEVICE, batch_size=BATCH_SIZE)
                learner.fit(train_loader, test_loader, DISABLE_FIRST_EPOCHS, save_last=SAVE_LAST)
            else:
                learner.fit(train_dataloader, test_dataloader, EPOCHS, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS, disable_xil_loss_first_n_epochs=DISABLE_FIRST_EPOCHS)

elif args.data == 'FMNIST':
    for n in range(5):
        util.seed_all(SEED[n])
        for meth in methods:
            if meth == 'RRR':
                train_dataloader, test_dataloader = decoy_mnist(fmnist=True, device=DEVICE, batch_size=BATCH_SIZE)      
                reg = 10
                loss_fn = RRRLoss(reg)
            elif meth == 'RBR':
                train_dataloader, test_dataloader = decoy_mnist(fmnist=True, device=DEVICE, batch_size=BATCH_SIZE)      
                reg = 1000000
                loss_fn = RBRLoss(args.reg)
            elif meth == 'RRR-G':
                train_dataloader, test_dataloader = decoy_mnist(fmnist=True, device=DEVICE, batch_size=BATCH_SIZE)      
                reg = 10
                loss_fn = RRRGradCamLoss(reg)
            elif meth == 'HINT':
                train_dataloader, test_dataloader = decoy_mnist(fmnist=True, hint_expl=True, device=DEVICE, batch_size=BATCH_SIZE)
                reg = 10
                loss_fn = HINTLoss(args.reg, last_conv_specified=True, upsample=True)
            elif meth == 'CE':
                train_dataloader, test_dataloader = decoy_mnist(fmnist=True, device=DEVICE, batch_size=BATCH_SIZE)      
                reg = None
                loss_fn = nn.CrossEntropyLoss()
            elif meth == 'CDEP':
                train_dataloader, test_dataloader = decoy_mnist(fmnist=True, device=DEVICE, batch_size=BATCH_SIZE)      
                reg = 1000  
                loss_fn = CDEPLoss(args.reg)
            print(f'{meth}-{n}')
            model = dnns.SimpleConvNet().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            modelname = f'SwitchOn--DecoyFMNIST-CNN-{meth}--seed={SEED[n]}--reg={args.reg}'
            learner = Learner(model, loss_fn, optimizer, DEVICE, modelname)
            if meth == 'CE':
                learner.fit(train_dataloader, test_dataloader, DISABLE_FIRST_EPOCHS, save_last=SAVE_LAST)
                train_loader, test_loader = decoy_mnist_CE_augmented(fmnist=True, device=DEVICE, batch_size=BATCH_SIZE)
                learner.fit(train_loader, test_loader, DISABLE_FIRST_EPOCHS, save_last=SAVE_LAST)
            else:
                learner.fit(train_dataloader, test_dataloader, EPOCHS, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS, disable_xil_loss_first_n_epochs=DISABLE_FIRST_EPOCHS)
