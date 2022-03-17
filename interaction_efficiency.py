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
EPOCHS = 50
BATCH_SIZE = 256
LR = 0.001
SAVE_LAST = True
VERBOSE_AFTER_N_EPOCHS = 2
DISABLE_FIRST_EPOCHS = 50
SEED = [1, 10, 100, 1000, 10000]
N_EXPLS = [25, 100, 200, 400, 800, 1600, 5000, 10000]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Using {DEVICE} device]")

parser = argparse.ArgumentParser(description='XIL EVAL')
parser.add_argument('-m', '--mode', default='RRR', type=str, choices=['RRR','RRR-G','HINT','CDEP','CE','RBR'],
                    help='Which XIL method to test?')
parser.add_argument('--data', default='MNIST', type=str, 
                    help='Which explainer to use?')
parser.add_argument('--reg', default=1000, type=int, 
                    help='Which explainer to use?')

args = parser.parse_args()
methods = [args.mode]

rtpt = RTPT(name_initials='FF', experiment_name='XIL_EVAL', max_iterations=len(SEED)*len(N_EXPLS))
rtpt.start()
# -

if args.data == 'MNIST':
    for n_expl in N_EXPLS:
        for n in range(len(SEED)):
            util.seed_all(SEED[n])
            for meth in methods:
                reg = 60000 / n_expl / BATCH_SIZE
                if meth == 'RRR':
                    train_dataloader, test_dataloader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE, n_expl=n_expl)      
                    reg *= 100
                    loss_fn = RRRLoss(reg)
                elif meth == 'RBR':
                    train_dataloader, test_dataloader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE, n_expl=n_expl)      
                    reg *= 100000
                    loss_fn = RBRLoss(reg)
                elif meth == 'RRR-G':
                    train_dataloader, test_dataloader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE, n_expl=n_expl)      
                    reg *= 1
                    loss_fn = RRRGradCamLoss(reg)
                elif meth == 'HINT':
                    train_dataloader, test_dataloader = decoy_mnist(hint_expl=True, device=DEVICE, batch_size=BATCH_SIZE, n_expl=n_expl)
                    #reg *= 100
                    reg = args.reg * reg
                    loss_fn = HINTLoss(reg, last_conv_specified=True, upsample=True)
                elif meth == 'CE':
                    train_dataloader, test_dataloader = decoy_mnist_CE_augmented(device=DEVICE, batch_size=BATCH_SIZE, n_instances=n_expl)      
                    reg *= None
                    loss_fn = nn.CrossEntropyLoss()
                elif meth == 'CDEP':
                    train_dataloader, test_dataloader = decoy_mnist(device=DEVICE, batch_size=BATCH_SIZE, n_expl=n_expl)      
                    #reg *= 1000
                    reg = args.reg * reg
                    loss_fn = CDEPLoss(reg)
                print(f'{meth}-{n}')
                rtpt.step()
                model = dnns.SimpleConvNet().to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                #modelname = f'IntEff-DecoyMNIST-CNN-{meth}--{n_expl}--seed={SEED[n]}'
                modelname = f'IntEff-DecoyMNIST-CNN-{meth}--{n_expl}--seed={SEED[n]}--reg={args.reg}'
                learner = Learner(model, loss_fn, optimizer, DEVICE, modelname)
                if meth == 'CE':
                    learner.fit(train_dataloader, test_dataloader, epochs=EPOCHS, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)
                else:
                    learner.fit_n_expl_shuffled_dataloader(train_dataloader, test_dataloader, epochs=EPOCHS, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)

if args.data == 'FMNIST':
    for n_expl in N_EXPLS:
        for n in range(len(SEED)):
            util.seed_all(SEED[n])
            for meth in methods:
                reg = 60000 / n_expl / BATCH_SIZE
                if meth == 'RRR':
                    train_dataloader, test_dataloader = decoy_mnist(fmnist=True, device=DEVICE, batch_size=BATCH_SIZE, n_expl=n_expl)      
                    reg *= 10
                    loss_fn = RRRLoss(reg)
                elif meth == 'RBR':
                    train_dataloader, test_dataloader = decoy_mnist(fmnist=True, device=DEVICE, batch_size=BATCH_SIZE, n_expl=n_expl)      
                    reg *= 1000000
                    loss_fn = RBRLoss(reg)
                elif meth == 'RRR-G':
                    train_dataloader, test_dataloader = decoy_mnist(fmnist=True, device=DEVICE, batch_size=BATCH_SIZE, n_expl=n_expl)      
                    reg *= 10
                    loss_fn = RRRGradCamLoss(reg)
                elif meth == 'HINT':
                    train_dataloader, test_dataloader = decoy_mnist(fmnist=True, hint_expl=True, device=DEVICE, batch_size=BATCH_SIZE, n_expl=n_expl)
                    reg *= 0.1
                    loss_fn = HINTLoss(reg, last_conv_specified=True, upsample=True)
                elif meth == 'CE':
                    train_dataloader, test_dataloader = decoy_mnist_CE_augmented(fmnist=True, device=DEVICE, batch_size=BATCH_SIZE, n_instances=n_expl)      
                    reg = None
                    loss_fn = nn.CrossEntropyLoss()
                elif meth == 'CDEP':
                    train_dataloader, test_dataloader = decoy_mnist(fmnist=True, device=DEVICE, batch_size=BATCH_SIZE, n_expl=n_expl)      
                    reg *= 1000  
                    loss_fn = CDEPLoss(reg)
                print(f'{meth}-{n}')
                model = dnns.SimpleConvNet().to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                modelname = f'IntEff--DecoyFMNIST-CNN-{meth}--{n_expl}--seed={SEED[n]}'
                learner = Learner(model, loss_fn, optimizer, DEVICE, modelname)
                if meth == 'CE':
                    learner.fit(train_dataloader, test_dataloader, epochs=EPOCHS, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)
                else:
                    learner.fit_n_expl_shuffled_dataloader(train_dataloader, test_dataloader, epochs=EPOCHS, save_last=SAVE_LAST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)
