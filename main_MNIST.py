"""Example calls to plot heatmaps and calculate WR on ISIC19."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils import data

from learner.models import dnns
from learner.learner import Learner
from data_store.datasets import decoy_mnist, decoy_mnist_CE_augmented, decoy_mnist_both, decoy_mnist_CE_combined, decoy_mnist_all
from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, RBRLoss, HINTLoss_IG, MixLoss1, MixLoss2, MixLoss3, \
    MixLoss4, MixLoss5, MixLoss6, MixLoss7, MixLoss8, MixLoss8_ext, MixLoss9, MixLoss11, MixLoss12, MixLoss13, MixLoss14, \
    MixLoss15, MixLoss16, MixLoss17, MixLoss18, MixLossGeneral, MixLossGeneralRevised
import util
import explainer
import matplotlib.pyplot as plt
import argparse
import os
from rtpt import RTPT

rtpt = RTPT(name_initials='EW', experiment_name='main_MNIST', max_iterations=256)

# +
# __import__("pdb").set_trace()
parser = argparse.ArgumentParser(description='XIL EVAL')
parser.add_argument('-m', '--mode', default='RRR', type=str, choices=['Vanilla','RRR','RRR-G','HINT','CDEP','CE','RBR', 'HINT_IG',\
                                                                      'Mix1', 'Mix2', 'Mix2ext', 'Mix3', 'Mix4', 'Mix5', 'Mix6', 'Mix7',\
                                                                      'Mix8', 'Mix8ext', 'Mix9', 'Mix11', 'Mix12', 'Mix13', 'Mix14',\
                                                                      'Mix15', 'Mix16', 'Mix17', 'Mix18', 'MixLoss'],
                    help='Which XIL method to test?')
parser.add_argument('--rrr', default=None, type=int)
parser.add_argument('--rbr', default=None, type=int)
parser.add_argument('--rrrg', default=None, type=int)
parser.add_argument('--hint', default=None, type=float)
parser.add_argument('--hint_ig', default=None, type=float)
parser.add_argument('--cdep', default=None, type=int)
parser.add_argument('--ce', default=False, type=bool)

parser.add_argument('--dataset', default='Mnist', type=str, choices=['Mnist','FMnist'],
                    help='Which dataset to use?')
parser.add_argument('--run', default=0, type=int,
                    help='Which Seed?')

args = parser.parse_args()
# -

# Get cpu or gpu device for training.
DEVICE = "cpu"
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
if args.dataset == 'Mnist':
    train_dataloader, test_dataloader = decoy_mnist(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
    if args.mode == 'Vanilla':
        args.reg = None
        args.mode = 'CEL'
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'RRR':
        if args.rrr is None:
            args.rrr = 10
        args.reg = args.rrr
        loss_fn = RRRLoss(args.reg)
    elif args.mode == 'RBR':
        if args.rbr is None:
            args.rbr = 100000
        args.reg = args.rbr
        loss_fn = RBRLoss(args.reg)
    elif args.mode == 'RRR-G':
        if args.rrrg is None:
            args.rrrg = 1
        args.reg = args.rrrg
        loss_fn = RRRGradCamLoss(args.reg)
        args.mode = 'RRRGradCAM'
    elif args.mode == 'HINT':
        train_dataloader, val_dataloader = decoy_mnist(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, \
                                       hint_expl=True)
        if args.hint is None:
            args.hint = 100
        args.reg = args.hint
        loss_fn = HINTLoss(args.reg, last_conv_specified=True, upsample=True, reduction='mean')
    elif args.mode == 'HINT_IG':
        train_dataloader, val_dataloader = decoy_mnist(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, \
                                       hint_expl=True)
        if args.hint_ig is None:
            args.hint_ig = 50000
        args.reg = args.hint_ig
        loss_fn = HINTLoss_IG(args.reg, reduction='mean')
    elif args.mode == 'CE':
        train_dataloader, val_dataloader = decoy_mnist_CE_augmented(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'CDEP':
        # args.reg = 1000000
        if args.cdep is None:
            args.cdep = 1000000
        args.reg = args.cdep
        loss_fn = CDEPLoss(args.reg)
    elif args.mode == 'Mix1':
        # Loss function combination of RRR, RBR, and RRRG
        args.reg = None
        loss_fn = MixLoss1(regrate_rrr=args.rrr, regrate_rbr=args.rbr, regrate_rrrg=args.rrrg)
    elif args.mode == 'Mix2':
        # Loss function combination of RRRG and HINT
        train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss2(regrate_rrrg=args.rrrg, regrate_hint=args.hint)
    elif args.mode == 'Mix3':
        # Loss function combination of RRR and CDEP
        args.reg = None
        loss_fn = MixLoss3(regrate_rrr=args.rrr, regrate_cdep=args.cdep)
    elif args.mode == 'Mix4':
        # Loss function combination of RRR and RBR
        args.reg = None
        loss_fn = MixLoss4(regrate_rrr=args.rrr, regrate_rbr=args.rbr)
    elif args.mode == 'Mix5':
        # Loss function combination of RBR and CDEP
        args.reg = None
        loss_fn = MixLoss5(regrate_rbr=args.rbr, regrate_cdep=args.cdep)
    elif args.mode == 'Mix6':
        # Loss function combination of RRRG and CDEP
        args.reg = None
        loss_fn = MixLoss6(regrate_rrrg=args.rrrg, regrate_cdep=args.cdep)
    elif args.mode == 'Mix7':
        # Loss function combination of CDEP and HINT
        train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss7(regrate_cdep=args.cdep, regrate_hint=args.hint)
    elif args.mode == 'Mix8':
        # Loss function combination of RRR and HINT
        train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss8(regrate_rrr=args.rrr, regrate_hint=args.hint)
    elif args.mode == 'Mix8ext':
        # Loss function combination of RRR and HINT_IG
        train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss8_ext(regrate_rrr=args.rrr, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix9':
        # Loss function combination of RBR and HINT
        train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss9(regrate_rbr=args.rbr, regrate_hint=args.hint)
    elif args.mode == 'Mix11':
        # Loss function combination of RBR and HINT
        train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss11(regrate_rbr=args.rbr, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix12':
        # Loss function combination of CDEP and HINT_IG
        train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss12(regrate_cdep=args.cdep, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix13':
        # Loss function combination of RRRG and HINT_IG
        train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss13(regrate_rrrg=args.rrrg, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix14':
        # Loss function combination of RRR and CE
        train_dataloader, val_dataloader = decoy_mnist_CE_combined(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rrr
        loss_fn = MixLoss14(args.reg)
    elif args.mode == 'Mix15':
        # Loss function combination of RBR and CE
        train_dataloader, val_dataloader = decoy_mnist_CE_combined(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rbr
        loss_fn = MixLoss15(args.reg)
    elif args.mode == 'Mix16':
        # Loss function combination of RRRG and CE
        train_dataloader, val_dataloader = decoy_mnist_CE_combined(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rrrg
        loss_fn = MixLoss16(args.reg)
    elif args.mode == 'Mix17':
        # Loss function combination of CDEP and CE
        train_dataloader, val_dataloader = decoy_mnist_CE_combined(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.cdep
        loss_fn = MixLoss17(args.reg)
    elif args.mode == 'Mix18':
        # Loss function combination of HINT and CE
        train_dataloader, val_dataloader = decoy_mnist_CE_combined(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, hint_expl=True)
        args.reg = args.hint
        loss_fn = MixLoss18(args.reg)
    elif args.mode == 'MixLoss':
        train_dataloader, val_dataloader = decoy_mnist_all(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, counter_ex=args.ce)
        args.reg = None

        regrate_rrr = None
        regrate_rbr = None
        regrate_rrrg = None
        regrate_cdep = None
        regrate_hint = None
        # regrate_hint_ig = None

        if args.rrr is not None:
            regrate_rrr = RRRLoss(args.rrr)
        if args.rbr is not None:
            regrate_rbr = RBRLoss(args.rbr)
        if args.rrrg is not None:
            regrate_rrrg = RRRGradCamLoss(args.rrrg)
        if args.cdep is not None:
            regrate_cdep = CDEPLoss(args.cdep)
        if args.hint is not None:
            regrate_hint = HINTLoss(args.hint, last_conv_specified=True, upsample=True)
        # if args.hint_ig is not None:
        #     regrate_hint_ig = args.hint_ig

        loss_fn = MixLossGeneralRevised(regrate_rrr=regrate_rrr, regrate_rbr=regrate_rbr, regrate_rrrg=regrate_rrrg, \
                                 regrate_cdep=regrate_cdep, regrate_hint=regrate_hint)

        
elif args.dataset == 'FMnist':
    train_dataloader, test_dataloader = decoy_mnist(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
    if args.mode == 'Vanilla':
        args.reg = None
        args.mode == 'CEL'
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'RRR':
        if args.rrr is None:
            args.rrr = 10
        args.reg = args.rrr
        loss_fn = RRRLoss(args.reg)
    elif args.mode == 'RBR':
        if args.rbr is None:
            args.rbr = 1000000
        args.reg = args.rbr
        loss_fn = RBRLoss(args.reg)
    elif args.mode == 'RRR-G':
        if args.rrrg is None:
            args.rrrg = 10
        args.reg = args.rrrg
        loss_fn = RRRGradCamLoss(args.reg)
        args.mode = 'RRRGradCAM'
    elif args.mode == 'HINT':
        train_dataloader, val_dataloader = decoy_mnist(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, \
                                               hint_expl=True)
        if args.hint is None:
            args.hint = 0.00001
        args.reg = args.hint
        loss_fn = HINTLoss(args.reg, last_conv_specified=True, upsample=True, reduction='mean')
    elif args.mode == 'HINT_IG':
        train_dataloader, val_dataloader = decoy_mnist(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, \
                                       hint_expl=True)
        if args.hint_ig is None:
            args.hint_ig = 90000
        args.reg = args.hint_ig
        loss_fn = HINTLoss_IG(args.reg, reduction='mean')
    elif args.mode == 'CE':
        train_dataloader, val_dataloader = decoy_mnist_CE_augmented(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'CDEP':
        if args.cdep is None:
            args.cdep = 2000000
        args.reg = args.cdep
        loss_fn = CDEPLoss(args.reg)
    elif args.mode == 'Mix1':
        # Loss function combination of RRR, RBR, and RRRG
        args.reg = None
        loss_fn = MixLoss1(regrate_rrr=args.rrr, regrate_rbr=args.rbr, regrate_rrrg=args.rrrg)
    elif args.mode == 'Mix2':
        # Loss function combination of RRRG + HINT
        train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss2(regrate_rrrg=args.rrrg, regrate_hint=args.hint)
    elif args.mode == 'Mix3':
        # Loss function combination of RRR and CDEP
        args.reg = None
        loss_fn = MixLoss3(regrate_rrr=args.rrr, regrate_cdep=args.cdep)
    elif args.mode == 'Mix4':
        # Loss function combination of RRR and RBR
        args.reg = None
        loss_fn = MixLoss4(regrate_rrr=args.rrr, regrate_rbr=args.rbr)
    elif args.mode == 'Mix5':
        # Loss function combination of RBR and CDEP
        args.reg = None
        loss_fn = MixLoss5(regrate_rbr=args.rbr, regrate_cdep=args.cdep)
    elif args.mode == 'Mix6':
        # Loss function combination of RRRG and CDEP
        args.reg = None
        loss_fn = MixLoss6(regrate_rrrg=args.rrrg, regrate_cdep=args.cdep)
    elif args.mode == 'Mix7':
        # Loss function combination of CDEP and HINT
        train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss7(regrate_cdep=args.cdep, regrate_hint=args.hint)
    elif args.mode == 'Mix8':
        # Loss function combination of RRR and HINT
        train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss8(regrate_rrr=args.rrr, regrate_hint=args.hint)
    elif args.mode == 'Mix8ext':
        # Loss function combination of RRR and HINT_IG
        train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss8_ext(regrate_rrr=args.rrr, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix9':
        # Loss function combination of RBR and HINT
        train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss9(regrate_rbr=args.rbr, regrate_hint=args.hint)
    elif args.mode == 'Mix11':
        # Loss function combination of RBR and HINT
        train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss11(regrate_rbr=args.rbr, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix12':
        # Loss function combination of CDEP and HINT_IG
        train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss12(regrate_cdep=args.cdep, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix13':
        # Loss function combination of RRRG and HINT_IG
        train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss13(regrate_rrrg=args.rrrg, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix14':
        # Loss function combination of RRR and CE
        train_dataloader, val_dataloader = decoy_mnist_CE_combined(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rrr
        loss_fn = MixLoss14(args.reg)
    elif args.mode == 'Mix15':
        # Loss function combination of RBR and CE
        train_dataloader, val_dataloader = decoy_mnist_CE_combined(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rbr
        loss_fn = MixLoss15(args.reg)
    elif args.mode == 'Mix16':
        # Loss function combination of RRRG and CE
        train_dataloader, val_dataloader = decoy_mnist_CE_combined(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rrrg
        loss_fn = MixLoss16(args.reg)
    elif args.mode == 'Mix17':
        # Loss function combination of CDEP and CE
        train_dataloader, val_dataloader = decoy_mnist_CE_combined(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.cdep
        loss_fn = MixLoss17(args.reg)
    elif args.mode == 'Mix18':
        # Loss function combination of HINT and CE
        train_dataloader, val_dataloader = decoy_mnist_CE_combined(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE,
                                                                   hint_expl=True)
        args.reg = args.hint
        loss_fn = MixLoss18(args.reg)
    elif args.mode == 'MixLoss':
        train_dataloader, val_dataloader = decoy_mnist_all(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, counter_ex=args.ce)
        args.reg = None

        regrate_rrr = None
        regrate_rbr = None
        regrate_rrrg = None
        regrate_cdep = None
        regrate_hint = None
        # regrate_hint_ig = None

        if args.rrr is not None:
            regrate_rrr = RRRLoss(args.rrr)
        if args.rbr is not None:
            regrate_rbr = RBRLoss(args.rbr)
        if args.rrrg is not None:
            regrate_rrrg = RRRGradCamLoss(args.rrrg)
        if args.cdep is not None:
            regrate_cdep = CDEPLoss(args.cdep)
        if args.hint is not None:
            regrate_hint = HINTLoss(args.hint, last_conv_specified=True, upsample=True)
        # if args.hint_ig is not None:
        #     regrate_hint_ig = args.hint_ig

        loss_fn = MixLossGeneralRevised(regrate_rrr=regrate_rrr, regrate_rbr=regrate_rbr, regrate_rrrg=regrate_rrrg, \
                                 regrate_cdep=regrate_cdep, regrate_hint=regrate_hint)

# -
# seed index
i = args.run

# if args.mode == 'Mix16':
#     util.seed_all(SEED2[i])
# else:
#     util.seed_all(SEED[i])

util.seed_all(SEED[i])

model = dnns.SimpleConvNet().to(DEVICE)
if args.mode == 'Mix1':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rrr},{args.rbr},{args.rrrg}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix2':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rrrg},{args.hint}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix3':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rrr},{args.cdep}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix4':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rrr},{args.rbr}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix5':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rbr},{args.cdep}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix6':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rrrg},{args.cdep}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix7':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.cdep},{args.hint}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix8':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rrr},{args.hint}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix8ext':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rrr},{args.hint_ig}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix9':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rbr},{args.hint}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix11':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rbr},{args.hint_ig}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix12':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.cdep},{args.hint_ig}--seed={SEED[i]}--run={i}'
elif args.mode == 'Mix13':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.rrrg},{args.hint_ig}--seed={SEED[i]}--run={i}'
elif args.mode == 'MixLoss':
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}'
    if args.rrr is not None:
        MODELNAME = MODELNAME + f'--rrr={args.rrr}'
    if args.rbr is not None:
        MODELNAME = MODELNAME + f'--rbr={args.rbr}'
    if args.rrrg is not None:
        MODELNAME = MODELNAME + f'--rrrg={args.rrrg}'
    if args.cdep is not None:
        MODELNAME = MODELNAME + f'--cdep={args.cdep}'
    if args.hint is not None:
        MODELNAME = MODELNAME + f'--hint={args.hint}'
    if args.ce:
        MODELNAME = MODELNAME + '--ce'
    MODELNAME = MODELNAME + f'--seed={SEED[i]}--run={i}'
else:
    MODELNAME = f'Decoy{args.dataset}-CNN-{args.mode}--reg={args.reg}--seed={SEED[i]}--run={i}'
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
learner = Learner(model, loss_fn, optimizer, DEVICE, MODELNAME)
learner.fit(train_dataloader, test_dataloader, EPOCHS, save_best=SAVE_BEST, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)
# avg0.append(learner.score(test_dataloader, nn.CrossEntropyLoss())[0])