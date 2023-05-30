########################
### Argument Parsing ###
########################

import argparse
parser = argparse.ArgumentParser(prog="Learner (F)MNIST")

parser.add_argument('--fmnist', default=False, action='store_true',
                    help='use the Fashion-MNIST instead of MNIST dataset')
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
        name_initials='EW',
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


def evaluate_model_on_explainers(trained_learners, test_loader, args, loss_config_string):

    avg_gradcam = []
    avg_ig = []
    avg_lime = []
    avg_saliency = []
    avg_ixg = []
    avg_deeplift = []
    avg_lrp = []
    avg_gbp = []
    avg9 = []

    dataset = 'F-MNIST' if args.fmnist else 'MNIST'

    for i, learner in enumerate(trained_learners):
        print(f'evaluating learner {i+1}/{len(trained_learners)}')

        if 'GradCAM' in args.explainer:
            # os.makedirs(
            #     f'output_images/{dataset}-expl/{loss_config_string}_grad/', exist_ok=True)
            # explainer.explain_with_captum('grad_cam', learner.model, test_loader, range(len(test_loader)),
            #                               next_to_each_other=False,
            #                               save_name=f'{dataset}-expl/{loss_config_string}_grad/{dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('grad_cam', test_loader, learner.model, mode='mean',
                                                     name=f'{loss_config_string}-grad', wr_name=learner.modelname + "--grad",
                                                     threshold=None, flags=False, device=DEVICE)
            avg_gradcam.append(explainer.quantify_wrong_reason('grad_cam', test_loader, learner.model, mode='mean',
                                                               name=f'{loss_config_string}-grad', wr_name=learner.modelname + "--grad",
                                                               threshold=thresh, flags=False, device=DEVICE))

        if 'Saliency' in args.explainer:
            # os.makedirs(
            #     f'output_images/{dataset}-expl/{loss_config_string}_saliency/', exist_ok=True)
            # explainer.explain_with_captum('saliency', learner.model, test_loader, range(len(test_loader)),
            #                               next_to_each_other=False,
            #                               save_name=f'{dataset}-expl/{loss_config_string}_saliency/{dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('saliency', test_loader, learner.model, mode='mean',
                                                     name=f'{loss_config_string}-saliency', wr_name=learner.modelname + "--saliency",
                                                     threshold=None, flags=False, device=DEVICE)
            avg_saliency.append(explainer.quantify_wrong_reason('saliency', test_loader, learner.model, mode='mean',
                                                                name=f'{loss_config_string}-saliency', wr_name=learner.modelname + "--saliency",
                                                                threshold=thresh, flags=False, device=DEVICE))

        if 'IxG' in args.explainer:
            # os.makedirs(
            #     f'output_images/{dataset}-expl/{loss_config_string}_input_x_gradient/', exist_ok=True)
            # explainer.explain_with_captum('input_x_gradient', learner.model, test_loader, range(len(test_loader)),
            #                               next_to_each_other=False,
            #                               save_name=f'{dataset}-expl/{loss_config_string}_input_x_gradient/{dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('input_x_gradient', test_loader, learner.model, mode='mean',
                                                     name=f'{loss_config_string}-input_x_gradient',
                                                     wr_name=learner.modelname + "--input_x_gradient",
                                                     threshold=None, flags=False, device=DEVICE)
            avg_ixg.append(explainer.quantify_wrong_reason('input_x_gradient', test_loader, learner.model, mode='mean',
                                                           name=f'{loss_config_string}-input_x_gradient',
                                                           wr_name=learner.modelname + "--input_x_gradient",
                                                           threshold=thresh, flags=False, device=DEVICE))

        if 'DeepLift' in args.explainer:
            # os.makedirs(
            #     f'output_images/{dataset}-expl/{loss_config_string}_deep_lift/', exist_ok=True)
            # explainer.explain_with_captum('deep_lift', learner.model, test_loader, range(len(test_loader)),
            #                               next_to_each_other=False,
            #                               save_name=f'{dataset}-expl/{loss_config_string}_deep_lift/{dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('deep_lift', test_loader, learner.model, mode='mean',
                                                     name=f'{loss_config_string}-deep_lift', wr_name=learner.modelname + "--deep_lift",
                                                     threshold=None, flags=False, device=DEVICE)
            avg_deeplift.append(explainer.quantify_wrong_reason('deep_lift', test_loader, learner.model, mode='mean',
                                                                name=f'{loss_config_string}-deep_lift', wr_name=learner.modelname + "--deep_lift",
                                                                threshold=thresh, flags=False, device=DEVICE))

        if 'LRP' in args.explainer:
            # os.makedirs(
            #     f'output_images/{dataset}-expl/{loss_config_string}_lrp/', exist_ok=True)
            # explainer.explain_with_captum('lrp', learner.model, test_loader, range(len(test_loader)),
            #                               next_to_each_other=False,
            #                               save_name=f'{dataset}-expl/{loss_config_string}_lrp/{dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('lrp', test_loader, learner.model, mode='mean',
                                                     name=f'{loss_config_string}-lrp', wr_name=learner.modelname + "--lrp",
                                                     threshold=None, flags=False, device=DEVICE)
            avg_lrp.append(explainer.quantify_wrong_reason('lrp', test_loader, learner.model, mode='mean',
                                                           name=f'{loss_config_string}-lrp', wr_name=learner.modelname + "--lrp",
                                                           threshold=thresh, flags=False, device=DEVICE))

        if 'GBP' in args.explainer:
            # os.makedirs(
            #     f'output_images/{dataset}-expl/{loss_config_string}_guided_backprop/', exist_ok=True)
            # explainer.explain_with_captum('guided_backprop', learner.model, test_loader, range(len(test_loader)),
            #                               next_to_each_other=False,
            #                               save_name=f'{dataset}-expl/{loss_config_string}_guided_backprop/{dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('guided_backprop', test_loader, learner.model, mode='mean',
                                                     name=f'{loss_config_string}-guided_backprop',
                                                     wr_name=learner.modelname + "--guided_backprop",
                                                     threshold=None, flags=False, device=DEVICE)
            avg_gbp.append(explainer.quantify_wrong_reason('guided_backprop', test_loader, learner.model, mode='mean',
                                                           name=f'{loss_config_string}-guided_backprop',
                                                           wr_name=learner.modelname + "--guided_backprop",
                                                           threshold=thresh, flags=False, device=DEVICE))

        # not implemented
        if 'IntGrad' in args.explainer:
            # os.makedirs(
            #     f'output_images/{dataset}-expl/{loss_config_string}_integrated_gradients/', exist_ok=True)
            # explainer.explain_with_captum('integrated_gradients', learner.model, test_loader, range(len(test_loader)),
            #                               next_to_each_other=False,
            #                               save_name=f'{dataset}-expl/{loss_config_string}_integrated_gradients/{dataset}-{loss_config_string}-test-wp-grad')
            thresh = explainer.quantify_wrong_reason('integrated_gradients', test_loader, learner.model, mode='mean',
                                                     name=f'{loss_config_string}-integrated_gradients',
                                                     wr_name=learner.modelname + "--integrated_gradients",
                                                     threshold=None, flags=False, device=DEVICE)
            avg9.append(explainer.quantify_wrong_reason('integrated_gradients', test_loader, learner.model, mode='mean',
                                                        name=f'{loss_config_string}-integrated_gradients',
                                                        wr_name=learner.modelname + "--integrated_gradients",
                                                        threshold=thresh, flags=False, device=DEVICE))

        if 'IG' in args.explainer:
            # os.makedirs(
            #     f'output_images/{dataset}-expl/{loss_config_string}_ig/', exist_ok=True)
            # explainer.explain_with_ig(learner.model, test_loader, range(len(test_loader)),
            #                           next_to_each_other=False,
            #                           save_name=f'{dataset}-expl/{loss_config_string}_ig/{dataset}-{loss_config_string}-test-wp-ig')
            thresh = explainer.quantify_wrong_reason('ig_ross', test_loader, learner.model, mode='mean',
                                                     name=f'{loss_config_string}-ig', wr_name=learner.modelname + "--ig",
                                                     threshold=None, flags=False, device=DEVICE)
            avg_ig.append(explainer.quantify_wrong_reason('ig_ross', test_loader, learner.model, mode='mean',
                                                          name=f'{loss_config_string}-ig', wr_name=learner.modelname + "--ig",
                                                          threshold=thresh, flags=False, device=DEVICE))

        if 'LIME' in args.explainer:
            # os.makedirs(
            #     f'output_images/{dataset}-expl/{loss_config_string}_lime/', exist_ok=True)
            # explainer.explain_with_lime(learner.model, test_loader, range(len(test_loader)),
            #                             next_to_each_other=False,
            #                             save_name=f'{dataset}-expl/{loss_config_string}_lime/{dataset}-{loss_config_string}-test-wp-lime')
            thresh = explainer.quantify_wrong_reason_lime(test_loader, learner.model, mode='mean',
                                                          name=f'{loss_config_string}-lime',
                                                          threshold=None, save_raw_attr=True, num_samples=2000, flags=False,
                                                          gray_images=True)
            avg_lime.append(explainer.quantify_wrong_reason_lime_preload(learner.model, mode='mean', name=f'{loss_config_string}-lime',
                                                                         threshold=thresh, device=DEVICE,
                                                                         batch_size=args.batch_size))

        # write results to file
        f = open(
            f"./output_wr_metric/{dataset}-{loss_config_string}.txt", "w")

        if 'IG' in args.explainer:
            f.write(f'IG P: mean:{np.mean(avg_ig)}, std:{np.std(avg_ig)}\n')
        if 'GradCAM' in args.explainer:
            f.write(
                f'Grad P: mean:{np.mean(avg_gradcam)}, std:{np.std(avg_gradcam)}\n')
        if 'LIME' in args.explainer:
            f.write(
                f'LIME P: mean:{np.mean(avg_lime)}, std:{np.std(avg_lime)}\n')
        if 'Saliency' in args.explainer:
            f.write(
                f'Saliency P: mean:{np.mean(avg_saliency)}, std:{np.std(avg_saliency)}\n')
        if 'IxG' in args.explainer:
            f.write(f'IxG P: mean:{np.mean(avg_ixg)}, std:{np.std(avg_ixg)}\n')
        if 'DeepLift' in args.explainer:
            f.write(
                f'DL P: mean:{np.mean(avg_deeplift)}, std:{np.std(avg_deeplift)}\n')
        if 'LRP' in args.explainer:
            f.write(f'LRP P: mean:{np.mean(avg_lrp)}, std:{np.std(avg_lrp)}\n')
        if 'GBP' in args.explainer:
            f.write(f'GBP P: mean:{np.mean(avg_gbp)}, std:{np.std(avg_gbp)}\n')
        if 'IntGrad' in args.explainer:
            f.write(f'IntGrad P: mean:{np.mean(avg9)}, std:{np.std(avg9)}\n')

        f.close()


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

reduced_training_size = None
if args.num_batches:
    reduced_training_size = args.batch_size * args.num_batches

train_loader, test_loader = decoy_mnist_all_revised(
    fmnist=args.fmnist,
    train_shuffle=True,
    device=DEVICE,
    batch_size=args.batch_size,
    generate_counterexamples=args.generate_counterexamples,
    reduced_training_size=reduced_training_size
)

######################
### Model Training ###
######################
trained_learners = train_model_on_losses(
    train_loader, test_loader, args, loss_config_string)

#########################
### Model Evaluation ####
#########################
evaluate_model_on_explainers(
    trained_learners, test_loader, args, loss_config_string)
