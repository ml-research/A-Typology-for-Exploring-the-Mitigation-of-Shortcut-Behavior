"""Main routine for training with ISIC19."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils import data
import logging

from learner.models import dnns
from learner.learner import Learner
from data_store.datasets import decoy_mnist, decoy_mnist_CE_augmented, isic_2019, isic_2019_hint
from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, RBRLoss
import util
import argparse


# +
# __import__("pdb").set_trace()
parser = argparse.ArgumentParser(description='XIL EVAL')
parser.add_argument('-m', '--mode', default='RRR', type=str, choices=['Vanilla','RRR','RRR-G','HINT','CDEP','CE','RBR'],
                    help='Which XIL method to test?')
parser.add_argument('--run', default=0, type=int, choices=[0,1,2,3,4],
                    help='Which XIL method to test?')

args = parser.parse_args()
DEVICE = "cuda"
LR = 0.001
SEED = [1, 10, 100, 1000, 10000]
BATCH_SIZE = 16
TRAIN_SHUFFLE = True
SAVE_LAST = True
SCHEDULER = True
EPOCHS = 50
VERBOSE_AFTER_N_EPOCHS = 1
# -

print("\nUsing {} device".format(DEVICE))

# +
############# Initalize dataset and dataloader
if args.mode == 'HINT':
    dataloaders, loss_weights = isic_2019_hint(batch_size=BATCH_SIZE, train_shuffle=TRAIN_SHUFFLE, all_hint=True)
elif args.mode == 'CE':
    dataloaders, loss_weights = isic_2019(batch_size=BATCH_SIZE, train_shuffle=TRAIN_SHUFFLE, ce_augment=True)
else:
    dataloaders, loss_weights = isic_2019(batch_size=BATCH_SIZE, train_shuffle=TRAIN_SHUFFLE)

train_dataloader, test_dataloader, test_no_patches = dataloaders["train"], dataloaders["test"],\
    dataloaders["test_no_patches"]

# +
########### Initalize model, loss and optimizer
base_criterion = nn.CrossEntropyLoss(weight=loss_weights.to(DEVICE))
   
i = args.run
logging.basicConfig(filename='isic_runs.log', level=logging.INFO, filemode='a', \
format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logging.info(f"\n\n--------RPEXP Train VGG16 (pretrained), XIL={args.mode}(0.1, clip=1.), base_criterion=CEL, dataset=ISIC2019")
logging.info(f"-HYPERPARAMS epochs= {EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}, optim=SGD(momentum=0.9), save_last={SAVE_LAST}, seeds={SEED[i]}, scheduler={SCHEDULER}, shuffle={TRAIN_SHUFFLE}")
util.seed_all(SEED[i])
MODELNAME = f'ISIC19-{args.mode}--seed={SEED[i]}--run={i}'
model = dnns.VGG16_pretrained_isic().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
if args.mode == 'Vanilla' or args.mode == 'CE':
    loss = base_criterion
    learner = Learner(model, loss, optimizer, DEVICE, MODELNAME, base_criterion=base_criterion)
    learner.fit_isic(train_dataloader, test_dataloader, EPOCHS, alternative_dataloader=test_no_patches, \
                     scheduler_=SCHEDULER, verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS, save_last=SAVE_LAST)
else:
    if args.mode == 'RRR':
        loss = RRRLoss(100, base_criterion=base_criterion, weight=loss_weights, rr_clipping=1.)
    elif args.mode == 'RBR':
        loss = RBRLoss(5, base_criterion=base_criterion, rr_clipping=1., weight=loss_weights)
    elif args.mode == 'RRR-G':
        loss = RRRGradCamLoss(0.1, base_criterion=base_criterion, reduction='mean', weight=loss_weights.to(DEVICE),
                              rr_clipping=1.)
    elif args.mode == 'HINT':
        loss = HINTLoss(1., base_criterion=base_criterion, reduction='none', weight=loss_weights.to(DEVICE))
    elif args.mode == 'CDEP':
        loss = CDEPLoss(10, base_criterion=base_criterion, weight=loss_weights, model_type='vgg')

    learner = Learner(model, loss, optimizer, DEVICE, MODELNAME, base_criterion=base_criterion)
    learner.fit_n_expl_shuffled_dataloader(train_dataloader, test_dataloader, EPOCHS, \
       alternative_dataloader=test_no_patches, save_last=SAVE_LAST,\
       verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS, scheduler_=SCHEDULER)
# -


############# Evaluate on test-P and test-NP set
print("TEST with patches: ")
learner.validation_statistics(test_dataloader, savename="-STATS-test-with-patches")
logging.info("Test set only patches DONE (see file in logfolder)")
print("Test no patches: ")
learner.validation_statistics(test_no_patches, savename="-STATS-test-no-patches")
logging.info("Test set no patches DONE (see file in logfolder)")
