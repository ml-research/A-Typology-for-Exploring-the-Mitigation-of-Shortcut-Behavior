"""Main routine for training with ISIC19."""
import argparse
parser = argparse.ArgumentParser(prog="Learner ISIC")

parser.add_argument('-b', '--batch-size', type=int, default=256)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-4)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('--dont-save-best-epoch', action='store_true')
parser.add_argument('-nt', '--num-threads', type=int, default=8)
parser.add_argument('-nb', '--num-batches', type=int,
                    help="will shrink dataset to number of batches")
parser.add_argument('-nr', '--no-restore', default=False,
                    action='store_true', help="do not try to load model from checkpoint")

parser.add_argument('-ce', '--generate-counterexamples', action='store_true')

# if loss function arg not provided -> value None -> loss func won't be used
# if loss function arg is provided without value -> use default 1. rate (rate will be evaluated during fit())
# if loss function arg is provided with  value -> use value
parser.add_argument('--rrr', const=1., nargs='?', type=float)
parser.add_argument('--rrr-gc', const=1., nargs='?', type=float)
parser.add_argument('--cdep', const=1., nargs='?', type=float)
parser.add_argument('--hint', const=1., nargs='?', type=float)
parser.add_argument('--hint-ig', const=1., nargs='?', type=float)
parser.add_argument('--rbr', const=1., nargs='?', type=float)

parser.add_argument('-nn', '--no-normalization', default=False, action='store_true',
                    help='disables normalization of right-reason loss functions')

parser.add_argument('-r', '--runs', type=int, default=[1, 2, 3, 4, 5], choices=[
                    1, 2, 3, 4, 5], nargs='+', help='specify runs to perform (each run uses its own seed)')

parser.add_argument('--explainer', default=[], type=str,
                    choices=['GradCAM', 'IG', 'LIME', 'Saliency',
                             'IxG', 'DeepLift', 'LRP', 'GBP', 'IntGrad'], nargs='+',
                    help='specifies explainers to use during evaluation')

args = parser.parse_args()


def train_model_on_losses(train_loader, test_loader, args, loss_config_string):

    # collect trained learners in list (gets iterated during eval)
    trained_learners = []

    # inform about run progress (instead of epochs)
    rtpt = RTPT(
        name_initials='FF',
        experiment_name='Learner',
        max_iterations=len(args.runs)
    )
    rtpt.start()

    for i, run_id in enumerate(args.runs):
        print(f"run {i+1}/{len(args.runs)}")

        util.seed_all(SEEDS[run_id-1])

        # generate unique and descriptive modelname
        dataset = 'F-MNIST' if args.fmnist else 'MNIST'
        MODELNAME = f'MLL_{dataset}{loss_config_string}_run={run_id}'

        untrained_model = dnns.SimpleConvNet().to(DEVICE)
        # untrained_model = torch.compile(untrained_model)
        optimizer = torch.optim.Adam(
            untrained_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        learner = Learner(
            untrained_model,
            optimizer,
            DEVICE,
            MODELNAME,
            restore_checkpoint=not args.no_restore
        )

        # learner.compare_original_and_revised_loss_functions(train_loader)
        # exit()

        # create dict with
        rr_loss_reg_rates = dict()
        if args.rrr:
            rr_loss_reg_rates['rrr'] = args.rrr
        if args.rrr_gc:
            rr_loss_reg_rates['rrr_gc'] = args.rrr_gc
        if args.cdep:
            rr_loss_reg_rates['cdep'] = args.cdep
        if args.hint:
            rr_loss_reg_rates['hint'] = args.hint
        if args.hint_ig:
            rr_loss_reg_rates['hint_ig'] = args.hint_ig
        if args.rbr:
            rr_loss_reg_rates['rbr'] = args.rbr

        learner.fit(
            train_loader,
            test_loader,
            args.epochs,
            rr_loss_reg_rates,

            normalize_loss_functions=not args.no_normalization,
            save_best_epoch=not args.dont_save_best_epoch,
        )

        trained_learners.append(learner)

        rtpt.step()

    return trained_learners


import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info(f'cli args: {args}')

# imports take lots of time -> do it after argparse
from rtpt import RTPT
import os
import util
from data_store.datasets import decoy_mnist_all_revised
from learner.learner import Learner
from learner.models import dnns
import torch
import explainer
import numpy as np

# args define training behaviour -> build config string shared by all models
loss_config_string = str()
if args.rrr:
    loss_config_string += f'_rrr={args.rrr}'
if args.rrr_gc:
    loss_config_string += f'_rrrg={args.rrr_gc}'
if args.cdep:
    loss_config_string += f'_cdep={args.cdep}'
if args.hint:
    loss_config_string += f'_hint={args.hint}'
if args.hint_ig:
    loss_config_string += f'_hintig={args.hint_ig}'
if args.rbr:
    loss_config_string += f'_rbr={args.rbr}'
if args.generate_counterexamples:
    loss_config_string += '_ce'

# different but pre-defined seed for each run
SEEDS = [1, 10, 100, 1000, 10000]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"DEVICE={DEVICE}")

torch.set_printoptions(linewidth=150)
torch.set_num_threads(args.num_threads)


###################
### Data Loader ###
###################

if args.hint:
    dataloaders, loss_weights = isic_2019_hint(batch_size=BATCH_SIZE, train_shuffle=True, all_hint=True)
elif args.generate_counterexamples:
    dataloaders, loss_weights = isic_2019(batch_size=BATCH_SIZE, train_shuffle=True, ce_augment=True)
else:
    dataloaders, loss_weights = isic_2019(batch_size=BATCH_SIZE, train_shuffle=True)

train_loader, test_loader, test_no_patches = dataloaders["train"], dataloaders["test"],\
    dataloaders["test_no_patches"]

######################
### Model Training ###
######################
trained_learners = train_model_on_losses(
    train_loader, test_loader, args, loss_config_string)
