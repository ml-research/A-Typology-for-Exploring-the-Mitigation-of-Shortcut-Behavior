"""Example calls to plot heatmaps and calculate WR on ISIC19."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils import data

from learner.models import dnns
from learner.learner import Learner
from data_store.datasets import decoy_mnist, decoy_mnist_CE_augmented, decoy_mnist_both, decoy_mnist_CE_combined, decoy_mnist_all
from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, HINTLoss_IG, RBRLoss, MixLoss1, MixLoss2, \
    MixLoss3, \
    MixLoss4, MixLoss5, MixLoss6, MixLoss7, MixLoss8, MixLoss8_ext, MixLoss9, MixLoss11, MixLoss12, MixLoss13, \
    MixLoss14, \
    MixLoss15, MixLoss16, MixLoss17, MixLoss18, MixLossGeneral, MixLossGeneralRevised
import util
import explainer
import matplotlib.pyplot as plt
import argparse
import os
from rtpt import RTPT

rtpt = RTPT(name_initials='RW', experiment_name='WR_MNIST', max_iterations=256)

# +
# __import__("pdb").set_trace()
parser = argparse.ArgumentParser(description='XIL EVAL')
parser.add_argument('-m', '--mode', default='RRR', type=str,
                    choices=['Vanilla', 'RRR', 'RRR-G', 'HINT', 'CDEP', 'CE', 'RBR', 'HINT_IG', \
                             'Mix1', 'Mix2', 'Mix3', 'Mix4', 'Mix5', 'Mix6', 'Mix7', \
                             'Mix8', 'Mix8ext', 'Mix9', 'Mix11', 'Mix12', 'Mix13', 'Mix14', \
                             'Mix15', 'Mix16', 'Mix17', 'Mix18', 'MixLoss'],
                    help='Which XIL method to test?')
parser.add_argument('--rrr', default=None, type=int)
parser.add_argument('--rbr', default=None, type=int)
parser.add_argument('--rrrg', default=None, type=int)
parser.add_argument('--hint', default=None, type=float)
parser.add_argument('--hint_ig', default=None, type=float)
parser.add_argument('--cdep', default=None, type=int)
parser.add_argument('--ce', default=False, type=bool)

parser.add_argument('--dataset', default='Mnist', type=str, choices=['Mnist', 'FMnist'],
                    help='Which dataset to use?')
parser.add_argument('--method', default='GradCAM IG LIME Saliency IxG DeepLift LRP GBP IntGrad', type=str,
                    choices=['GradCAM', 'IG', 'LIME', 'Saliency', \
                             'IxG', 'DeepLift', 'LRP', 'GBP', 'IntGrad'], nargs='+',
                    help='Which explainer to use?')
parser.add_argument('--run', default=0, type=int,
                    help='Which seed?')
parser.add_argument('--batch', default=256, type=int)
parser.add_argument('--retrain', default=False, type=bool)

args = parser.parse_args()
# -

# Get cpu or gpu device for training.
DEVICE = "cuda"
SEED = [1, 10, 100, 1000, 10000]
SHUFFLE = False
# BATCH_SIZE = 256
BATCH_SIZE = args.batch
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 50
SAVE_BEST = True
VERBOSE_AFTER_N_EPOCHS = 2
RETRAIN = args.retrain

print("\nUsing {} device".format(DEVICE))

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
        # train_dataloader, val_dataloader = decoy_mnist(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, \
        #                                hint_expl=True)
        if args.hint is None:
            args.hint = 100
        args.reg = args.hint
        loss_fn = HINTLoss(args.reg, last_conv_specified=True, upsample=True, reduction='mean')
    elif args.mode == 'HINT_IG':
        # train_dataloader, val_dataloader = decoy_mnist(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, \
        #                                hint_expl=True)
        if args.hint_ig is None:
            args.hint_ig = 50000
        args.reg = args.hint_ig
        loss_fn = HINTLoss_IG(args.reg, reduction='mean')
    elif args.mode == 'CE':
        # train_dataloader, val_dataloader = decoy_mnist_CE_augmented(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'CDEP':
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
        # train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
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
        # train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss7(regrate_cdep=args.cdep, regrate_hint=args.hint)
    elif args.mode == 'Mix8':
        # Loss function combination of RRR and HINT
        # train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss8(regrate_rrr=args.rrr, regrate_hint=args.hint)
    elif args.mode == 'Mix8ext':
        # Loss function combination of RRR and HINT_IG
        # train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss8_ext(regrate_rrr=args.rrr, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix9':
        # Loss function combination of RBR and HINT
        # train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss9(regrate_rbr=args.rbr, regrate_hint=args.hint)
    elif args.mode == 'Mix11':
        # Loss function combination of RBR and HINT
        # train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss11(regrate_rbr=args.rbr, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix12':
        # Loss function combination of CDEP and HINT_IG
        # train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss12(regrate_cdep=args.cdep, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix13':
        # Loss function combination of RRRG and HINT_IG
        # train_dataloader, val_dataloader = decoy_mnist_both(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss13(regrate_rrrg=args.rrrg, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix14':
        # Loss function combination of RRR and CE
        # train_dataloader, val_dataloader = decoy_mnist_CE_combined(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rrr
        loss_fn = MixLoss14(args.reg)
    elif args.mode == 'Mix15':
        # Loss function combination of RBR and CE
        # train_dataloader, val_dataloader = decoy_mnist_CE_combined(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rbr
        loss_fn = MixLoss15(args.reg)
    elif args.mode == 'Mix16':
        # Loss function combination of RRRG and CE
        # train_dataloader, val_dataloader = decoy_mnist_CE_combined(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rrrg
        loss_fn = MixLoss16(args.reg)
    elif args.mode == 'Mix17':
        # Loss function combination of CDEP and CE
        # train_dataloader, val_dataloader = decoy_mnist_CE_combined(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.cdep
        loss_fn = MixLoss17(args.reg)
    elif args.mode == 'Mix18':
        # Loss function combination of HINT and CE
        # train_dataloader, val_dataloader = decoy_mnist_CE_combined(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, hint_expl=True)
        args.reg = args.hint
        loss_fn = MixLoss18(args.reg)
    elif args.mode == 'MixLoss':
        # train_dataloader, val_dataloader = decoy_mnist_all(train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE,
        #                                                    counter_ex=args.ce)
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
            regrate_hint = HINTLoss(args.hint)
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
        if args.hint is None:
            args.hint = 0.00001
        args.reg = args.hint
        loss_fn = HINTLoss(args.reg, last_conv_specified=True, upsample=True)
    elif args.mode == 'HINT_IG':
        # train_dataloader, val_dataloader = decoy_mnist(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, \
        #                                hint_expl=True)
        if args.hint_ig is None:
            args.hint_ig = 90000
        args.reg = args.hint_ig
        loss_fn = HINTLoss_IG(args.reg, reduction='mean')
    elif args.mode == 'CE':
        # train_dataloader, val_dataloader = decoy_mnist_CE_augmented(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE,
        #                                                             batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = nn.CrossEntropyLoss()
    elif args.mode == 'CDEP':
        args.reg = 2000000
        loss_fn = CDEPLoss(args.reg)
    elif args.mode == 'Mix1':
        # Loss function combination of RRR, RBR, and RRRG
        args.reg = None
        loss_fn = MixLoss1(regrate_rrr=args.rrr, regrate_rbr=args.rbr, regrate_rrrg=args.rrrg)
    elif args.mode == 'Mix2':
        # Loss function combination of RRRG + HINT
        # train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
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
        # train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss7(regrate_cdep=args.cdep, regrate_hint=args.hint)
    elif args.mode == 'Mix8':
        # Loss function combination of RRR and HINT
        # train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss8(regrate_rrr=args.rrr, regrate_hint=args.hint)
    elif args.mode == 'Mix8ext':
        # Loss function combination of RRR and HINT_IG
        # train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss8_ext(regrate_rrr=args.rrr, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix9':
        # Loss function combination of RBR and HINT
        # train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss9(regrate_rbr=args.rbr, regrate_hint=args.hint)
    elif args.mode == 'Mix11':
        # Loss function combination of RBR and HINT
        # train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss11(regrate_rbr=args.rbr, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix12':
        # Loss function combination of CDEP and HINT_IG
        # train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss12(regrate_cdep=args.cdep, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix13':
        # Loss function combination of RRRG and HINT_IG
        # train_dataloader, val_dataloader = decoy_mnist_both(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = None
        loss_fn = MixLoss13(regrate_rrrg=args.rrrg, regrate_hint_ig=args.hint_ig)
    elif args.mode == 'Mix14':
        # Loss function combination of RRR and CE
        # train_dataloader, val_dataloader = decoy_mnist_CE_combined(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rrr
        loss_fn = MixLoss14(args.reg)
    elif args.mode == 'Mix15':
        # Loss function combination of RBR and CE
        # train_dataloader, val_dataloader = decoy_mnist_CE_combined(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rbr
        loss_fn = MixLoss15(args.reg)
    elif args.mode == 'Mix16':
        # Loss function combination of RRRG and CE
        # train_dataloader, val_dataloader = decoy_mnist_CE_combined(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.rrrg
        loss_fn = MixLoss16(args.reg)
    elif args.mode == 'Mix17':
        # Loss function combination of CDEP and CE
        # train_dataloader, val_dataloader = decoy_mnist_CE_combined(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.cdep
        loss_fn = MixLoss17(args.reg)
    elif args.mode == 'Mix18':
        # Loss function combination of HINT and CE
        # train_dataloader, val_dataloader = decoy_mnist_CE_combined(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE)
        args.reg = args.hint
        loss_fn = MixLoss18(args.reg)
    elif args.mode == 'MixLoss':
        # train_dataloader, val_dataloader = decoy_mnist_all(fmnist=True, train_shuffle=SHUFFLE, device=DEVICE,
        #                                                    batch_size=BATCH_SIZE, counter_ex=args.ce)
        args.reg = None

        # TO DO : gives if for loss function
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
            regrate_hint = HINTLoss(args.hint)
        # if args.hint_ig is not None:
        #     regrate_hint_ig = args.hint_ig

        loss_fn = MixLossGeneralRevised(regrate_rrr=regrate_rrr, regrate_rbr=regrate_rbr, regrate_rrrg=regrate_rrrg, \
                                 regrate_cdep=regrate_cdep, regrate_hint=regrate_hint)


# +
avg1 = []
avg2 = []
avg3 = []
avg4 = []
avg5 = []
avg6 = []
avg7 = []
avg8 = []
avg9 = []

for i in range(5):
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
    learner = Learner(model, loss_fn, optimizer, DEVICE, MODELNAME, load=True)

    if RETRAIN:
        test_dataloader = train_dataloader

    if 'GradCAM' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_grad/', exist_ok=True)
        # explainer.explain_with_captum('grad_cam', learner.model, test_dataloader, range(len(test_dataloader)), \
        #                               next_to_each_other=False,
        #                               save_name=f'{args.dataset}-expl/{args.mode}_grad/{args.dataset}-{args.mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('grad_cam', test_dataloader, learner.model, mode='mean',
                                                 name=f'{args.mode}-grad', wr_name=MODELNAME + "--grad", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg1.append(explainer.quantify_wrong_reason('grad_cam', test_dataloader, learner.model, mode='mean',
                                                    name=f'{args.mode}-grad', wr_name=MODELNAME + "--grad", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'Saliency' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_saliency/', exist_ok=True)
        # explainer.explain_with_captum('saliency', learner.model, test_dataloader, range(len(test_dataloader)), \
        #                               next_to_each_other=False,
        #                               save_name=f'{args.dataset}-expl/{args.mode}_saliency/{args.dataset}-{args.mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('saliency', test_dataloader, learner.model, mode='mean',
                                                 name=f'{args.mode}-saliency', wr_name=MODELNAME + "--saliency", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg4.append(explainer.quantify_wrong_reason('saliency', test_dataloader, learner.model, mode='mean',
                                                    name=f'{args.mode}-saliency', wr_name=MODELNAME + "--saliency", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'IxG' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_input_x_gradient/', exist_ok=True)
        # explainer.explain_with_captum('input_x_gradient', learner.model, test_dataloader, range(len(test_dataloader)), \
        #                               next_to_each_other=False,
        #                               save_name=f'{args.dataset}-expl/{args.mode}_input_x_gradient/{args.dataset}-{args.mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('input_x_gradient', test_dataloader, learner.model, mode='mean',
                                                 name=f'{args.mode}-input_x_gradient',
                                                 wr_name=MODELNAME + "--input_x_gradient", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg5.append(explainer.quantify_wrong_reason('input_x_gradient', test_dataloader, learner.model, mode='mean',
                                                    name=f'{args.mode}-input_x_gradient',
                                                    wr_name=MODELNAME + "--input_x_gradient", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'DeepLift' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_deep_lift/', exist_ok=True)
        # explainer.explain_with_captum('deep_lift', learner.model, test_dataloader, range(len(test_dataloader)), \
        #                               next_to_each_other=False,
        #                               save_name=f'{args.dataset}-expl/{args.mode}_deep_lift/{args.dataset}-{args.mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('deep_lift', test_dataloader, learner.model, mode='mean',
                                                 name=f'{args.mode}-deep_lift', wr_name=MODELNAME + "--deep_lift", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg6.append(explainer.quantify_wrong_reason('deep_lift', test_dataloader, learner.model, mode='mean',
                                                    name=f'{args.mode}-deep_lift', wr_name=MODELNAME + "--deep_lift", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'LRP' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_lrp/', exist_ok=True)
        # explainer.explain_with_captum('lrp', learner.model, test_dataloader, range(len(test_dataloader)), \
        #                               next_to_each_other=False,
        #                               save_name=f'{args.dataset}-expl/{args.mode}_lrp/{args.dataset}-{args.mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('lrp', test_dataloader, learner.model, mode='mean',
                                                 name=f'{args.mode}-lrp', wr_name=MODELNAME + "--lrp", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg7.append(explainer.quantify_wrong_reason('lrp', test_dataloader, learner.model, mode='mean',
                                                    name=f'{args.mode}-lrp', wr_name=MODELNAME + "--lrp", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'GBP' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_guided_backprop/', exist_ok=True)
        # explainer.explain_with_captum('guided_backprop', learner.model, test_dataloader, range(len(test_dataloader)), \
        #                               next_to_each_other=False,
        #                               save_name=f'{args.dataset}-expl/{args.mode}_guided_backprop/{args.dataset}-{args.mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('guided_backprop', test_dataloader, learner.model, mode='mean',
                                                 name=f'{args.mode}-guided_backprop',
                                                 wr_name=MODELNAME + "--guided_backprop", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg8.append(explainer.quantify_wrong_reason('guided_backprop', test_dataloader, learner.model, mode='mean',
                                                    name=f'{args.mode}-guided_backprop',
                                                    wr_name=MODELNAME + "--guided_backprop", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'IntGrad' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_integrated_gradient/', exist_ok=True)
        # explainer.explain_with_captum('integrated_gradient', learner.model, test_dataloader, range(len(test_dataloader)), \
        #                               next_to_each_other=False,
        #                               save_name=f'{args.dataset}-expl/{args.mode}_integrated_gradient/{args.dataset}-{args.mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('integrated_gradient', test_dataloader, learner.model, mode='mean',
                                                 name=f'{args.mode}-integrated_gradient',
                                                 wr_name=MODELNAME + "--integrated_gradient", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg9.append(explainer.quantify_wrong_reason('integrated_gradient', test_dataloader, learner.model, mode='mean',
                                                    name=f'{args.mode}-integrated_gradient',
                                                    wr_name=MODELNAME + "--integrated_gradient", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'IG' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_ig/', exist_ok=True)
        # explainer.explain_with_ig(learner.model, test_dataloader, range(len(test_dataloader)), \
        #                           next_to_each_other=False,
        #                           save_name=f'{args.dataset}-expl/{args.mode}_ig/{args.dataset}-{args.mode}-test-wp-ig')
        thresh = explainer.quantify_wrong_reason('ig_ross', test_dataloader, learner.model, mode='mean',
                                                 name=f'{args.mode}-ig', wr_name=MODELNAME + "--ig", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg2.append(explainer.quantify_wrong_reason('ig_ross', test_dataloader, learner.model, mode='mean',
                                                    name=f'{args.mode}-ig', wr_name=MODELNAME + "--ig", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'LIME' in args.method:
        os.makedirs(f'output_images/{args.dataset}-expl/{args.mode}_lime/', exist_ok=True)
        # explainer.explain_with_lime(learner.model, test_dataloader, range(len(test_dataloader)), \
        #                             next_to_each_other=False,
        #                             save_name=f'{args.dataset}-expl/{args.mode}_lime/{args.dataset}-{args.mode}-test-wp-lime')
        thresh = explainer.quantify_wrong_reason_lime(test_dataloader, learner.model, mode='mean',
                                                      name=f'{args.mode}-lime', \
                                                      threshold=None, save_raw_attr=True, num_samples=1000, flags=False,
                                                      gray_images=True)
        avg3.append(explainer.quantify_wrong_reason_lime_preload(learner.model, mode='mean', name=f'{args.mode}-lime', \
                                                                 threshold=thresh, device=DEVICE,
                                                                 batch_size=BATCH_SIZE))

if args.mode == 'Mix1':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rrr}-{args.rbr}-{args.rrrg}.txt", "w")
elif args.mode == 'Mix2':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rrrg}-{args.hint}.txt", "w")
elif args.mode == 'Mix3':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rrr}-{args.cdep}.txt", "w")
elif args.mode == 'Mix4':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rrr}-{args.rbr}.txt", "w")
elif args.mode == 'Mix5':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rbr}-{args.cdep}.txt", "w")
elif args.mode == 'Mix6':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rrrg}-{args.cdep}.txt", "w")
elif args.mode == 'Mix7':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.cdep}-{args.hint}.txt", "w")
elif args.mode == 'Mix8':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rrr}-{args.hint}.txt", "w")
elif args.mode == 'Mix8ext':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rrr}-{args.hint_ig}.txt", "w")
elif args.mode == 'Mix9':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rbr}-{args.hint}.txt", "w")
elif args.mode == 'Mix11':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rbr}-{args.hint_ig}.txt", "w")
elif args.mode == 'Mix12':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.cdep}-{args.hint_ig}.txt", "w")
elif args.mode == 'Mix13':
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.rrrg}-{args.hint_ig}.txt", "w")
elif args.mode == 'MixLoss':
    temp = ""
    if args.rrr is not None:
        temp = temp + f'--rrr={args.rrr}'
    if args.rbr is not None:
        temp = temp + f'--rbr={args.rbr}'
    if args.rrrg is not None:
        temp = temp + f'--rrrg={args.rrrg}'
    if args.cdep is not None:
        temp = temp + f'--cdep={args.cdep}'
    if args.hint is not None:
        temp = temp + f'--hint={args.hint}'
    if args.ce:
        temp = temp + '--ce'
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}{temp}.txt", "w")
else:
    f = open(f"./output_wr_metric/{args.dataset}-{args.mode}-{args.reg}.txt", "w")
f.write(f'Grad P: mean:{np.mean(avg1)}, std:{np.std(avg1)}\n'
        f'IG P: mean:{np.mean(avg2)}, std:{np.std(avg2)}\n'
        f'LIME P: mean:{np.mean(avg3)}, std:{np.std(avg3)}\n'
        f'Saliency P: mean:{np.mean(avg4)}, std:{np.std(avg4)}\n'
        f'IxG P: mean:{np.mean(avg5)}, std:{np.std(avg5)}\n'
        f'DL P: mean:{np.mean(avg6)}, std:{np.std(avg6)}\n'
        f'LRP P: mean:{np.mean(avg7)}, std:{np.std(avg7)}\n'
        f'GBP P: mean:{np.mean(avg8)}, std:{np.std(avg8)}\n'
        f'IntGrad P: mean:{np.mean(avg9)}, std:{np.std(avg9)}\n')
f.close()
