import os
import util
from data_store.datasets import decoy_mnist_all_revised
from learner.learner import Learner
from learner.models import dnns
import torch
import argparse
import logging
import explainer
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(prog="MultiLoss (F)MNIST")
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-f', '--fmnist', action='store_true')
parser.add_argument('-b', '--batch-size', type=int, default=250)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-4)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-sb', '--save-best-epoch', action='store_true')
parser.add_argument('-t', '--num-threads', type=int, default=5)
parser.add_argument('-nce', '--no-counterexamples', action='store_true')
parser.add_argument('-l', '--load-model', action='store_true') # load previously trained model

parser.add_argument('--rrr-rr', type=int)
parser.add_argument('--rrr-gc-rr', type=int)
parser.add_argument('--cdep-rr', type=int)
parser.add_argument('--hint-rr', type=int)
parser.add_argument('--hint-ig-rr', type=int)
parser.add_argument('--rbr-rr', type=int)

args = parser.parse_args()

logging.info(f'cli args: {args}')

MODELNAME = 'MultiLoss(F)MNIST'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Compute DEVICE={DEVICE}")

from rtpt import RTPT
rtpt = RTPT(
    name_initials='EW',
    experiment_name=MODELNAME, 
    max_iterations=args.epochs
)

torch.set_printoptions(linewidth=150)
torch.set_num_threads(args.num_threads)
util.seed_all(args.seed)

train_loader, test_loader = decoy_mnist_all_revised(
    fmnist=args.fmnist,
    train_shuffle=True,
    device=DEVICE,
    batch_size=args.batch_size,
    generate_counterexamples=not args.no_counterexamples,
    #reduced_training_size=args.batch_size * 10
)

model = dnns.SimpleConvNet().to(DEVICE)
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=args.learning_rate, 
    weight_decay=args.weight_decay
)


learner = Learner(
    model,
    optimizer,
    DEVICE,
    MODELNAME,
    load=args.load_model
)

if not args.load_model:
    learner.fit(
        train_loader,
        test_loader,
        args.epochs,
        rtpt,
        save_best_epoch=args.save_best_epoch,

        loss_rrr_regularizer_rate=args.rrr_rr,
        loss_rrr_gc_regularizer_rate=args.rrr_gc_rr,
        loss_cdep_regularizer_rate=args.cdep_rr,
        loss_hint_regularizer_rate=args.hint_rr,
        loss_hint_ig_regularizer_rate=args.hint_ig_rr,
        loss_rbr_regularizer_rate=args.rbr_rr,
    )
else:
    logging.info('Model was loaded from file, skipping training')


dataset = 'fmnist' if args.fmnist else 'mnist'
mode = f'rrr={args.rrr_rr},rrr-gc={args.rrr_gc_rr},cdep={args.cdep_rr},hint={args.hint_rr},hint-ig={args.hint_ig_rr},rbr={args.rbr_rr}'

os.makedirs(f'output_images/{dataset}-expl/{mode}_saliency/', exist_ok=True)
explainer.explain_with_captum('saliency', learner.model, test_loader, range(len(test_loader)), \
                              next_to_each_other=False,)
                            #   save_name=f'{dataset}-expl/{mode}_saliency/{dataset}-{mode}-test-wp-grad')
# thresh = explainer.quantify_wrong_reason('saliency', test_dataloader, learner.model, mode='mean',
#                                             name=f'{args.mode}-saliency', wr_name=MODELNAME + "--saliency", \
#                                             threshold=None, flags=False, device=DEVICE)
# avg4.append(explainer.quantify_wrong_reason('saliency', test_dataloader, learner.model, mode='mean',
#                                             name=f'{args.mode}-saliency', wr_name=MODELNAME + "--saliency", \
#                                             threshold=thresh, flags=False, device=DEVICE))