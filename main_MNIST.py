"""Example calls to plot heatmaps and calculate WR on ISIC19."""
import torch
from learner.models import dnns
from learner.learner import Learner
from data_store.datasets import decoy_mnist_all_revised

import util
from rtpt import RTPT

torch.set_printoptions(linewidth=150)

SEED = [1, 10, 100, 1000, 10000]
SHUFFLE = True
BATCH_SIZE = 250
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
EPOCHS = 50
SAVE_BEST_EPOCH = True
VERBOSE_AFTER_N_EPOCHS = 2
MODELNAME = 'SuperMODEL3000'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\nUsing {} device".format(DEVICE))

rtpt = RTPT(name_initials='EW',
            experiment_name='main_MNIST', max_iterations=EPOCHS)

# TODO allow seed selection
util.seed_all(SEED[0])

model = dnns.SimpleConvNet().to(DEVICE)

train_loader, test_loader = decoy_mnist_all_revised(
    fmnist=True, train_shuffle=SHUFFLE, device=DEVICE, batch_size=BATCH_SIZE, generate_counterexamples=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
learner = Learner(model, optimizer, DEVICE, MODELNAME,
                           loss_rrr_weight=1.0,
                           loss_rrr_gc_weight=torch.Tensor([1.] * 10),
                           loss_cdep_weight=torch.Tensor([1.] * 10),
                           loss_hint_weight=torch.Tensor([1.] * 10),
                           loss_rbr_weight=torch.Tensor([1.] * 10),
                           loss_ce=True
                           )
learner.fit(train_loader, test_loader, EPOCHS, save_best_epoch=SAVE_BEST_EPOCH,
            verbose_after_n_epochs=VERBOSE_AFTER_N_EPOCHS)
# avg0.append(learner.score(test_dataloader, nn.CrossEntropyLoss())[0])
