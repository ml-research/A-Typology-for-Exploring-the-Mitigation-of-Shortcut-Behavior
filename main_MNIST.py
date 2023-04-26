########################
### Argument Parsing ###
########################

import argparse
parser = argparse.ArgumentParser(prog="Learner (F)MNIST")

parser.add_argument('-d', '--dataset', default='mnist', type=str, choices=['mnist', 'fmnist'])
parser.add_argument('-b', '--batch-size', type=int, default=250)
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
parser.add_argument('-wd', '--weight-decay', type=float, default=1e-4)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('--dont-save-best-epoch', action='store_true')
#parser.add_argument('-t', '--num-threads', type=int, default=5)
parser.add_argument('-rt', '--reduced-train-set', type=int, help="if set will shrink train set to x * batch_size")

parser.add_argument('--ce', action='store_true')
parser.add_argument('--rrr', const=0., nargs='?', type=float)
parser.add_argument('--rrr-gc', const=0., nargs='?', type=float)
parser.add_argument('--cdep', const=0., nargs='?', type=float)
parser.add_argument('--hint', const=0., nargs='?', type=float)
parser.add_argument('--hint-ig', const=0., nargs='?', type=float)
parser.add_argument('--rbr', const=0., nargs='?', type=float)

parser.add_argument('-r', '--runs', type=int, default=[1, 2, 3, 4, 5], choices=[1, 2, 3, 4, 5], nargs='+', help='Which runs to perform (each run has a different seed)?')

parser.add_argument('--explainer', default='GradCAM IG LIME Saliency IxG DeepLift LRP GBP IntGrad', type=str,
                    choices=['GradCAM', 'IG', 'LIME', 'Saliency', \
                             'IxG', 'DeepLift', 'LRP', 'GBP', 'IntGrad'], nargs='+',
                    help='Which explainers to use?')

args = parser.parse_args()



def train_learners(train_loader, test_loader, args):

    # collect trained learners in list (gets iterated during eval)
    trained_learners = []

    # inform about run progress (instead of epochs)
    rtpt = RTPT(
        name_initials='EW',
        experiment_name='Learner', 
        max_iterations=len(args.runs)
    )
    rtpt.start()

    for i, run_id in enumerate(args.runs):
        print(f"run {i+1}/{len(args.runs)}")

        util.seed_all(SEEDS[run_id-1])

        # generate unique and descriptive modelname
        MODELNAME = f'MLL_{args.dataset}{loss_config_string}_run={run_id}'

        untrained_model = dnns.SimpleConvNet().to(DEVICE)
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
        )

        # if loss func specified AND reg-rate are set via cli args we use provided values
        # if loss func specified via cli args, but no reg-rate is provided, we evaluate best value on training set
        # if loss func not specified, we don't use it
        # TODO
        args.rrr, args.rrr_gc, args.cdep, args.hint, args.hint_ig, args.rbr = learner.evaluate_regularization_rates(train_loader)


        learner.fit(
            train_loader,
            test_loader,
            args.epochs,
            save_best_epoch=not args.dont_save_best_epoch,

            loss_rrr_regularizer_rate=args.rrr,
            loss_rrr_gc_regularizer_rate=args.rrr_gc,
            loss_cdep_regularizer_rate=args.cdep,
            loss_hint_regularizer_rate=args.hint,
            loss_hint_ig_regularizer_rate=args.hint_ig,
            loss_rbr_regularizer_rate=args.rbr,
        )

        trained_learners.append(learner)

        rtpt.step()

    return trained_learners


def evaluate_on_explainers(trained_learners, test_loader, args):

    avg_gradcam = []
    avg_ig = []
    avg_lime = []
    avg_saliency = []
    avg_ixg = []
    avg_deeplift = []
    avg_lrp = []
    avg_gbp = []
    avg9 = []

    for i, learner in enumerate(trained_learners):
        print(f'evaluating learner {i+1}/{len(trained_learners)}')

        if 'GradCAM' in args.explainer:
            os.makedirs(f'output_images/{args.dataset}-expl/{loss_config_string}_grad/', exist_ok=True)
            explainer.explain_with_captum('grad_cam', learner.model, test_loader, range(len(test_loader)), \
                                        next_to_each_other=False,
                                        save_name=f'{args.dataset}-expl/{loss_config_string}_grad/{args.dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('grad_cam', test_loader, learner.model, mode='mean',
                                                    name=f'{loss_config_string}-grad', wr_name=learner.modelname + "--grad", \
                                                    threshold=None, flags=False, device=DEVICE)
            avg_gradcam.append(explainer.quantify_wrong_reason('grad_cam', test_loader, learner.model, mode='mean',
                                                        name=f'{loss_config_string}-grad', wr_name=learner.modelname + "--grad", \
                                                        threshold=thresh, flags=False, device=DEVICE))

        if 'Saliency' in args.explainer:
            os.makedirs(f'output_images/{args.dataset}-expl/{loss_config_string}_saliency/', exist_ok=True)
            explainer.explain_with_captum('saliency', learner.model, test_loader, range(len(test_loader)), \
                                        next_to_each_other=False,
                                        save_name=f'{args.dataset}-expl/{loss_config_string}_saliency/{args.dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('saliency', test_loader, learner.model, mode='mean',
                                                    name=f'{loss_config_string}-saliency', wr_name=learner.modelname + "--saliency", \
                                                    threshold=None, flags=False, device=DEVICE)
            avg_saliency.append(explainer.quantify_wrong_reason('saliency', test_loader, learner.model, mode='mean',
                                                        name=f'{loss_config_string}-saliency', wr_name=learner.modelname + "--saliency", \
                                                        threshold=thresh, flags=False, device=DEVICE))

        if 'IxG' in args.explainer:
            os.makedirs(f'output_images/{args.dataset}-expl/{loss_config_string}_input_x_gradient/', exist_ok=True)
            explainer.explain_with_captum('input_x_gradient', learner.model, test_loader, range(len(test_loader)), \
                                        next_to_each_other=False,
                                        save_name=f'{args.dataset}-expl/{loss_config_string}_input_x_gradient/{args.dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('input_x_gradient', test_loader, learner.model, mode='mean',
                                                    name=f'{loss_config_string}-input_x_gradient',
                                                    wr_name=learner.modelname + "--input_x_gradient", \
                                                    threshold=None, flags=False, device=DEVICE)
            avg_ixg.append(explainer.quantify_wrong_reason('input_x_gradient', test_loader, learner.model, mode='mean',
                                                        name=f'{loss_config_string}-input_x_gradient',
                                                        wr_name=learner.modelname + "--input_x_gradient", \
                                                        threshold=thresh, flags=False, device=DEVICE))

        if 'DeepLift' in args.explainer:
            os.makedirs(f'output_images/{args.dataset}-expl/{loss_config_string}_deep_lift/', exist_ok=True)
            explainer.explain_with_captum('deep_lift', learner.model, test_loader, range(len(test_loader)), \
                                        next_to_each_other=False,
                                        save_name=f'{args.dataset}-expl/{loss_config_string}_deep_lift/{args.dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('deep_lift', test_loader, learner.model, mode='mean',
                                                    name=f'{loss_config_string}-deep_lift', wr_name=learner.modelname + "--deep_lift", \
                                                    threshold=None, flags=False, device=DEVICE)
            avg_deeplift.append(explainer.quantify_wrong_reason('deep_lift', test_loader, learner.model, mode='mean',
                                                        name=f'{loss_config_string}-deep_lift', wr_name=learner.modelname + "--deep_lift", \
                                                        threshold=thresh, flags=False, device=DEVICE))

        if 'LRP' in args.explainer:
            os.makedirs(f'output_images/{args.dataset}-expl/{loss_config_string}_lrp/', exist_ok=True)
            explainer.explain_with_captum('lrp', learner.model, test_loader, range(len(test_loader)), \
                                        next_to_each_other=False,
                                        save_name=f'{args.dataset}-expl/{loss_config_string}_lrp/{args.dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('lrp', test_loader, learner.model, mode='mean',
                                                    name=f'{loss_config_string}-lrp', wr_name=learner.modelname + "--lrp", \
                                                    threshold=None, flags=False, device=DEVICE)
            avg_lrp.append(explainer.quantify_wrong_reason('lrp', test_loader, learner.model, mode='mean',
                                                        name=f'{loss_config_string}-lrp', wr_name=learner.modelname + "--lrp", \
                                                        threshold=thresh, flags=False, device=DEVICE))

        if 'GBP' in args.explainer:
            os.makedirs(f'output_images/{args.dataset}-expl/{loss_config_string}_guided_backprop/', exist_ok=True)
            explainer.explain_with_captum('guided_backprop', learner.model, test_loader, range(len(test_loader)), \
                                        next_to_each_other=False,
                                        save_name=f'{args.dataset}-expl/{loss_config_string}_guided_backprop/{args.dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('guided_backprop', test_loader, learner.model, mode='mean',
                                                    name=f'{loss_config_string}-guided_backprop',
                                                    wr_name=learner.modelname + "--guided_backprop", \
                                                    threshold=None, flags=False, device=DEVICE)
            avg_gbp.append(explainer.quantify_wrong_reason('guided_backprop', test_loader, learner.model, mode='mean',
                                                        name=f'{loss_config_string}-guided_backprop',
                                                        wr_name=learner.modelname + "--guided_backprop", \
                                                        threshold=thresh, flags=False, device=DEVICE))

        #not implemented
        if 'IntGrad' in args.explainer:
            os.makedirs(f'output_images/{args.dataset}-expl/{loss_config_string}_integrated_gradients/', exist_ok=True)
            explainer.explain_with_captum('integrated_gradients', learner.model, test_loader, range(len(test_loader)), \
                                        next_to_each_other=False,
                                        save_name=f'{args.dataset}-expl/{loss_config_string}_integrated_gradients/{args.dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('integrated_gradients', test_loader, learner.model, mode='mean',
                                                    name=f'{loss_config_string}-integrated_gradients',
                                                    wr_name=learner.modelname + "--integrated_gradients", \
                                                    threshold=None, flags=False, device=DEVICE)
            avg9.append(explainer.quantify_wrong_reason('integrated_gradients', test_loader, learner.model, mode='mean',
                                                        name=f'{loss_config_string}-integrated_gradients',
                                                        wr_name=learner.modelname + "--integrated_gradients", \
                                                        threshold=thresh, flags=False, device=DEVICE))

        if 'IG' in args.explainer:
            os.makedirs(f'output_images/{args.dataset}-expl/{loss_config_string}_ig/', exist_ok=True)
            explainer.explain_with_ig(learner.model, test_loader, range(len(test_loader)), \
                                    next_to_each_other=False,
                                    save_name=f'{args.dataset}-expl/{loss_config_string}_ig/{args.dataset}-{loss_config_string}-test-wp-ig')
            thresh = explainer.quantify_wrong_reason('ig_ross', test_loader, learner.model, mode='mean',
                                                    name=f'{loss_config_string}-ig', wr_name=learner.modelname + "--ig", \
                                                    threshold=None, flags=False, device=DEVICE)
            avg_ig.append(explainer.quantify_wrong_reason('ig_ross', test_loader, learner.model, mode='mean',
                                                        name=f'{loss_config_string}-ig', wr_name=learner.modelname + "--ig", \
                                                        threshold=thresh, flags=False, device=DEVICE))

        if 'LIME' in args.explainer:
            print("lime start")
            os.makedirs(f'output_images/{args.dataset}-expl/{loss_config_string}_lime/', exist_ok=True)
            explainer.explain_with_lime(learner.model, test_loader, range(len(test_loader)), \
                                        next_to_each_other=False,
                                        save_name=f'{args.dataset}-expl/{loss_config_string}_lime/{args.dataset}-{loss_config_string}-test-wp-lime')
            thresh = explainer.quantify_wrong_reason_lime(test_loader, learner.model, mode='mean',
                                                        name=f'{loss_config_string}-lime', \
                                                        threshold=None, save_raw_attr=True, num_samples=2000, flags=False,
                                                        gray_images=True)
            print("done thresh")
            avg_lime.append(explainer.quantify_wrong_reason_lime_preload(learner.model, mode='mean', name=f'{loss_config_string}-lime', \
                                                                    threshold=thresh, device=DEVICE,
                                                                    batch_size=args.batch_size))
            print("done append")



        # write results to file
        f = open(f"./output_wr_metric/{args.dataset}-{loss_config_string}.txt", "w")
        
        if 'IG' in args.explainer:
            f.write(f'IG P: mean:{np.mean(avg_ig)}, std:{np.std(avg_ig)}\n')
        if 'GradCAM' in args.explainer:
            f.write(f'Grad P: mean:{np.mean(avg_gradcam)}, std:{np.std(avg_gradcam)}\n')
        if 'LIME' in args.explainer:
            f.write(f'LIME P: mean:{np.mean(avg_lime)}, std:{np.std(avg_lime)}\n')
        if 'Saliency' in args.explainer:
            f.write(f'Saliency P: mean:{np.mean(avg_saliency)}, std:{np.std(avg_saliency)}\n')
        if 'IxG' in args.explainer:
            f.write(f'IxG P: mean:{np.mean(avg_ixg)}, std:{np.std(avg_ixg)}\n')
        if 'DeepLift' in args.explainer:
            f.write(f'DL P: mean:{np.mean(avg_deeplift)}, std:{np.std(avg_deeplift)}\n')
        if 'LRP' in args.explainer:
            f.write(f'LRP P: mean:{np.mean(avg_lrp)}, std:{np.std(avg_lrp)}\n')
        if 'GBP' in args.explainer:
            f.write(f'GBP P: mean:{np.mean(avg_gbp)}, std:{np.std(avg_gbp)}\n')
        if 'IntGrad' in args.explainer:
            f.write(f'IntGrad P: mean:{np.mean(avg9)}, std:{np.std(avg9)}\n')

        f.close()


import logging
logging.basicConfig(level=logging.INFO)
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
if args.ce:
    loss_config_string += '_ce'

# different but pre-defined seed for each run
SEEDS = [1, 10, 100, 1000, 10000]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"DEVICE={DEVICE}")

torch.set_printoptions(linewidth=150)
#torch.set_num_threads(args.num_threads)


###################
### Data Loader ###
###################

reduced_training_size = None
if args.reduced_train_set:
    reduced_training_size = args.batch_size * args.reduced_train_set

train_loader, test_loader = decoy_mnist_all_revised(
    fmnist=args.dataset,
    train_shuffle=True,
    device=DEVICE,
    batch_size=args.batch_size,
    generate_counterexamples=args.ce,
    reduced_training_size=reduced_training_size
)

######################
### Model Training ###
######################
trained_learners = train_learners(train_loader, test_loader, args)

#########################
### Model Evaluation ####
#########################
evaluate_on_explainers(trained_learners, test_loader, args)