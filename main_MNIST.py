import os
import util
from data_store.datasets import decoy_mnist_all_revised
from learner.learner import Learner
from learner.models import dnns
import torch
import argparse
import logging
import explainer
import numpy as np
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

parser.add_argument('--rrr', type=int)
parser.add_argument('--rrrg', type=int)
parser.add_argument('--cdep', type=int)
parser.add_argument('--hint', type=int)
parser.add_argument('--hint_ig', type=int)
parser.add_argument('--rbr', type=int)

parser.add_argument('--method', default='GradCAM IG LIME Saliency IxG DeepLift LRP GBP IntGrad', type=str,
                    choices=['GradCAM', 'IG', 'LIME', 'Saliency', \
                             'IxG', 'DeepLift', 'LRP', 'GBP', 'IntGrad'], nargs='+',
                    help='Which explainer to use?')

args = parser.parse_args()

logging.info(f'cli args: {args}')

mode = f'rrr={args.rrr},rrrg={args.rrrg},cdep={args.cdep},hint={args.hint},hint-ig={args.hint_ig},rbr={args.rbr}'
MODELNAME = 'MultiLoss(F)MNIST,' + mode

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

        loss_rrr_regularizer_rate=args.rrr,
        loss_rrr_gc_regularizer_rate=args.rrrg,
        loss_cdep_regularizer_rate=args.cdep,
        loss_hint_regularizer_rate=args.hint,
        loss_hint_ig_regularizer_rate=args.hint_ig,
        loss_rbr_regularizer_rate=args.rbr,
    )
else:
    logging.info('Model was loaded from file, skipping training')


dataset = 'fmnist' if args.fmnist else 'mnist'

# +
avg_gradcam = []
avg_ig = []
avg_lime = []
avg_saliency = []
avg_ixg = []
avg_deeplift = []
avg_lrp = []
avg_gbp = []
avg9 = []

test_dataloader = test_loader
BATCH_SIZE = args.batch_size

for i in range(5):
    print(f"in loop iter={i}")
    if 'GradCAM' in args.method:
        os.makedirs(f'output_images/{dataset}-expl/{mode}_grad/', exist_ok=True)
        explainer.explain_with_captum('grad_cam', learner.model, test_dataloader, range(len(test_dataloader)), \
                                      next_to_each_other=False,
                                      save_name=f'{dataset}-expl/{mode}_grad/{dataset}-{mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('grad_cam', test_dataloader, learner.model, mode='mean',
                                                 name=f'{mode}-grad', wr_name=MODELNAME + "--grad", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg_gradcam.append(explainer.quantify_wrong_reason('grad_cam', test_dataloader, learner.model, mode='mean',
                                                    name=f'{mode}-grad', wr_name=MODELNAME + "--grad", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'Saliency' in args.method:
        os.makedirs(f'output_images/{dataset}-expl/{mode}_saliency/', exist_ok=True)
        explainer.explain_with_captum('saliency', learner.model, test_dataloader, range(len(test_dataloader)), \
                                      next_to_each_other=False,
                                      save_name=f'{dataset}-expl/{mode}_saliency/{dataset}-{mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('saliency', test_dataloader, learner.model, mode='mean',
                                                 name=f'{mode}-saliency', wr_name=MODELNAME + "--saliency", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg_saliency.append(explainer.quantify_wrong_reason('saliency', test_dataloader, learner.model, mode='mean',
                                                    name=f'{mode}-saliency', wr_name=MODELNAME + "--saliency", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'IxG' in args.method:
        os.makedirs(f'output_images/{dataset}-expl/{mode}_input_x_gradient/', exist_ok=True)
        explainer.explain_with_captum('input_x_gradient', learner.model, test_dataloader, range(len(test_dataloader)), \
                                      next_to_each_other=False,
                                      save_name=f'{dataset}-expl/{mode}_input_x_gradient/{dataset}-{mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('input_x_gradient', test_dataloader, learner.model, mode='mean',
                                                 name=f'{mode}-input_x_gradient',
                                                 wr_name=MODELNAME + "--input_x_gradient", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg_ixg.append(explainer.quantify_wrong_reason('input_x_gradient', test_dataloader, learner.model, mode='mean',
                                                    name=f'{mode}-input_x_gradient',
                                                    wr_name=MODELNAME + "--input_x_gradient", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'DeepLift' in args.method:
        os.makedirs(f'output_images/{dataset}-expl/{mode}_deep_lift/', exist_ok=True)
        explainer.explain_with_captum('deep_lift', learner.model, test_dataloader, range(len(test_dataloader)), \
                                      next_to_each_other=False,
                                      save_name=f'{dataset}-expl/{mode}_deep_lift/{dataset}-{mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('deep_lift', test_dataloader, learner.model, mode='mean',
                                                 name=f'{mode}-deep_lift', wr_name=MODELNAME + "--deep_lift", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg_deeplift.append(explainer.quantify_wrong_reason('deep_lift', test_dataloader, learner.model, mode='mean',
                                                    name=f'{mode}-deep_lift', wr_name=MODELNAME + "--deep_lift", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'LRP' in args.method:
        os.makedirs(f'output_images/{dataset}-expl/{mode}_lrp/', exist_ok=True)
        explainer.explain_with_captum('lrp', learner.model, test_dataloader, range(len(test_dataloader)), \
                                      next_to_each_other=False,
                                      save_name=f'{dataset}-expl/{mode}_lrp/{dataset}-{mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('lrp', test_dataloader, learner.model, mode='mean',
                                                 name=f'{mode}-lrp', wr_name=MODELNAME + "--lrp", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg_lrp.append(explainer.quantify_wrong_reason('lrp', test_dataloader, learner.model, mode='mean',
                                                    name=f'{mode}-lrp', wr_name=MODELNAME + "--lrp", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'GBP' in args.method:
        os.makedirs(f'output_images/{dataset}-expl/{mode}_guided_backprop/', exist_ok=True)
        explainer.explain_with_captum('guided_backprop', learner.model, test_dataloader, range(len(test_dataloader)), \
                                      next_to_each_other=False,
                                      save_name=f'{dataset}-expl/{mode}_guided_backprop/{dataset}-{mode}-test-wp-grad')
        thresh = explainer.quantify_wrong_reason('guided_backprop', test_dataloader, learner.model, mode='mean',
                                                 name=f'{mode}-guided_backprop',
                                                 wr_name=MODELNAME + "--guided_backprop", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg_gbp.append(explainer.quantify_wrong_reason('guided_backprop', test_dataloader, learner.model, mode='mean',
                                                    name=f'{mode}-guided_backprop',
                                                    wr_name=MODELNAME + "--guided_backprop", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    # not implemented
    # if 'IntGrad' in args.method:
    #     os.makedirs(f'output_images/{dataset}-expl/{mode}_integrated_gradient/', exist_ok=True)
    #     explainer.explain_with_captum('integrated_gradient', learner.model, test_dataloader, range(len(test_dataloader)), \
    #                                   next_to_each_other=False,
    #                                   save_name=f'{dataset}-expl/{mode}_integrated_gradient/{dataset}-{mode}-test-wp-grad')
    #     thresh = explainer.quantify_wrong_reason('integrated_gradient', test_dataloader, learner.model, mode='mean',
    #                                              name=f'{mode}-integrated_gradient',
    #                                              wr_name=MODELNAME + "--integrated_gradient", \
    #                                              threshold=None, flags=False, device=DEVICE)
    #     avg9.append(explainer.quantify_wrong_reason('integrated_gradient', test_dataloader, learner.model, mode='mean',
    #                                                 name=f'{mode}-integrated_gradient',
    #                                                 wr_name=MODELNAME + "--integrated_gradient", \
    #                                                 threshold=thresh, flags=False, device=DEVICE))

    if 'IG' in args.method:
        os.makedirs(f'output_images/{dataset}-expl/{mode}_ig/', exist_ok=True)
        explainer.explain_with_ig(learner.model, test_dataloader, range(len(test_dataloader)), \
                                  next_to_each_other=False,
                                  save_name=f'{dataset}-expl/{mode}_ig/{dataset}-{mode}-test-wp-ig')
        thresh = explainer.quantify_wrong_reason('ig_ross', test_dataloader, learner.model, mode='mean',
                                                 name=f'{mode}-ig', wr_name=MODELNAME + "--ig", \
                                                 threshold=None, flags=False, device=DEVICE)
        avg_ig.append(explainer.quantify_wrong_reason('ig_ross', test_dataloader, learner.model, mode='mean',
                                                    name=f'{mode}-ig', wr_name=MODELNAME + "--ig", \
                                                    threshold=thresh, flags=False, device=DEVICE))

    if 'LIME' in args.method:
        print("lime start")
        os.makedirs(f'output_images/{dataset}-expl/{mode}_lime/', exist_ok=True)
        explainer.explain_with_lime(learner.model, test_dataloader, range(len(test_dataloader)), \
                                    next_to_each_other=False,
                                    save_name=f'{dataset}-expl/{mode}_lime/{dataset}-{mode}-test-wp-lime')
        thresh = explainer.quantify_wrong_reason_lime(test_dataloader, learner.model, mode='mean',
                                                      name=f'{mode}-lime', \
                                                      threshold=None, save_raw_attr=True, num_samples=1000, flags=False,
                                                      gray_images=True)
        print("done thresh")
        avg_lime.append(explainer.quantify_wrong_reason_lime_preload(learner.model, mode='mean', name=f'{mode}-lime', \
                                                                 threshold=thresh, device=DEVICE,
                                                                 batch_size=BATCH_SIZE))
        print("done append")

    f = open(f"./output_wr_metric/{dataset}-{mode}.txt", "w")
    f.write(f'Grad P: mean:{np.mean(avg_gradcam)}, std:{np.std(avg_gradcam)}\n'
            f'IG P: mean:{np.mean(avg_ig)}, std:{np.std(avg_ig)}\n'
            f'LIME P: mean:{np.mean(avg_lime)}, std:{np.std(avg_lime)}\n'
            f'Saliency P: mean:{np.mean(avg_saliency)}, std:{np.std(avg_saliency)}\n'
            f'IxG P: mean:{np.mean(avg_ixg)}, std:{np.std(avg_ixg)}\n'
            f'DL P: mean:{np.mean(avg_deeplift)}, std:{np.std(avg_deeplift)}\n'
            f'LRP P: mean:{np.mean(avg_lrp)}, std:{np.std(avg_lrp)}\n'
            f'GBP P: mean:{np.mean(avg_gbp)}, std:{np.std(avg_gbp)}\n'
            f'IntGrad P: mean:{np.mean(avg9)}, std:{np.std(avg9)}\n')
    f.close()
