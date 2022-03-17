"""Example calls to plot heatmaps and calculate WR on ISIC19."""
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.utils import data
import os

from learner.models import dnns
from learner.learner import Learner
from data_store.datasets import isic_2019, isic_2019_hint
from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, RBRLoss
import util
import explainer
import matplotlib.pyplot as plt
import argparse
from rtpt import RTPT


# +
# __import__("pdb").set_trace()
parser = argparse.ArgumentParser(description='XIL EVAL')
parser.add_argument('-m', '--mode', default='RRR', type=str, choices=['Vanilla','RRR','RRR-G','HINT','CDEP','CE','RBR'],
                    help='Which XIL method to test?')
parser.add_argument('--method', default='GradCAM IG LIME', type=str, choices=['GradCAM','IG','LIME'], nargs='+', 
                    help='Which explainer to use?')
parser.add_argument('--run', default=0, type=int,
                    help='Which Seed?')

args = parser.parse_args()
# -

# Get cpu or gpu device for training.
DEVICE = "cuda"
LR = 0.001
BATCH_SIZE = 16
SEED = [1, 10, 100, 1000, 10000]
TRAIN_SHUFFLE = True
SAVE_LAST = True
SCHEDULER = True
EPOCHS = 50
VERBOSE_AFTER_N_EPOCHS = 1

print("\nUsing {} device".format(DEVICE))

# +
############# Initalize dataset and dataloader
dataloaders, loss_weights = isic_2019(batch_size=BATCH_SIZE, train_shuffle=True)#, number_nc=800, number_c=100)
train_dataloader, test_dataloader, test_no_patches = dataloaders["train"], dataloaders["test"], \
dataloaders["test_no_patches"]
base_criterion = nn.CrossEntropyLoss(weight=loss_weights.to(DEVICE))

if args.mode == 'Vanilla' or args.mode == 'CE':
    loss = base_criterion
else:
    if args.mode == 'RRR':
        loss = RRRLoss(100, base_criterion=base_criterion, rr_clipping=1.0, weight=loss_weights)
    elif args.mode == 'RBR':
        loss = RBRLoss(100, base_criterion=base_criterion, rr_clipping=1.0, weight=loss_weights)
    elif args.mode == 'RRR-G':
        loss = RRRGradCamLoss(0.1, base_criterion=base_criterion, reduction='mean', weight=loss_weights.to(DEVICE),
                              rr_clipping=1.)
    elif args.mode == 'HINT':
        loss = HINTLoss(1., base_criterion=base_criterion, reduction='none', weight=loss_weights.to(DEVICE))
    elif args.mode == 'CDEP':
        loss = CDEPLoss(10, base_criterion=base_criterion, weight=loss_weights, model_type='vgg')

# +
#### Load pretrained model from model_store
avg1, avg2, avg3= [], [], []
i = args.run
model = dnns.VGG16_pretrained_isic().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
util.seed_all(SEED[i])
MODELNAME = f'ISIC19-{args.mode}--seed={SEED[i]}--run={i}'
learner = Learner(model, loss, optimizer, DEVICE, MODELNAME, base_criterion=base_criterion, load=True)
rtpt.step()

if 'GradCAM' in args.method:
    os.makedirs(f'output_images/ISIC19-expl/{args.mode}_gradcam/', exist_ok=True)
    explainer.explain_with_captum_one_by_one('grad_cam', learner.model, test_dataloader, \
                 next_to_each_other=False, save_name=f'ISIC19-expl/{args.mode}_gradcam/ISIC19-{args.mode}-test-np-gradcam', device=DEVICE)
    thresh = explainer.quantify_wrong_reason('grad_cam', test_dataloader, learner.model, DEVICE, "name", \
    threshold=None, mode='mean', flags=True)
    avg1.append(explainer.quantify_wrong_reason('grad_cam', test_dataloader, learner.model, DEVICE, "name", \
    threshold=thresh, mode='mean', flags=True))

if 'IG' in args.method:
    os.makedirs(f'output_images/ISIC19-expl/{args.mode}_ig/', exist_ok=True)
    explainer.explain_with_ig_one_by_one(learner.model, test_dataloader, \
          next_to_each_other=False, save_name=f'ISIC19-expl/{args.mode}_ig/ISIC19-{args.mode}-test-np-ig', device=DEVICE)
    thresh = explainer.quantify_wrong_reason('ig_ross', test_dataloader, learner.model, DEVICE, "name", \
    threshold=None, mode='mean', flags=True)
    avg2.append(explainer.quantify_wrong_reason('ig_ross', test_dataloader, learner.model, DEVICE, "name", \
    threshold=thresh, mode='mean', flags=True))

if 'LIME' in args.method:
    explainer.explain_with_lime_one_by_one(learner.model, test_dataloader, \
         next_to_each_other=False, save_name=f'ISIC19-expl/{args.mode}_lime/ISIC19-{args.mode}-test-np-lime')  
    os.makedirs(f'output_images/ISIC19-expl/{args.mode}_lime/', exist_ok=True)
    thresh = explainer.quantify_wrong_reason_lime(test_dataloader, learner.model, mode='mean', name=f'{args.mode}-lime', \
        threshold=None, save_raw_attr=True, num_samples=1000, flags=True)
    avg3.append(explainer.quantify_wrong_reason_lime_preload(learner.model, mode='mean', name=f'{args.mode}-lime', \
        threshold=thresh, device=DEVICE, batch_size=BATCH_SIZE))


f = open(f"./output_wr_metric/ISIC19-{args.mode}-{args.method}-{args.run}.txt", "w")
f.write(f'Grad: {avg1}\n '
        f'IG: {avg2}\n '
        f'LIME:{avg3}\n '
        f'Grad: mean:{np.mean(avg1)}, std:{np.std(avg1)}\n '
        f'IG: mean:{np.mean(avg2)}, std:{np.std(avg2)}\n '
        f'LIME: mean:{np.mean(avg3)}, std:{np.std(avg3)}\n '
f.close()
