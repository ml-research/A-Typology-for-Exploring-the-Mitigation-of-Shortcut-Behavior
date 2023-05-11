import logging
import time
import uuid
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, HINTLoss_IG, RBRLoss
from datetime import datetime
from collections import defaultdict

class Learner:
    """
    Learner that can be configured to train on multiple weighted loss-functions simultaneously
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        device: str,
        modelname: str,
        loss_function_right_answer=F.cross_entropy,
    ):

        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.modelname = modelname
        self.loss_function_right_answer = loss_function_right_answer

        self.n_trained_epochs = 0

        # model automatically tries to load checkpoint from a previous run
        self.load_from_checkpoint()

    def calculate_normalization_rates(self, train_loader, loss_function_keys):
        """
        Will calculate regularization rates for all `loss_function_keys` on `train_loader`
        """

        # instantiate all loss functions in advance (may not all be used)
        loss_functions_rr = dict()
        loss_functions_rr['rrr'] = RRRLoss(
            normalization_rate=1., regularization_rate=1.)
        loss_functions_rr['rrr_gc'] = RRRGradCamLoss(
            normalization_rate=1., regularization_rate=1.)
        loss_functions_rr['cdep'] = CDEPLoss(
            normalization_rate=1., regularization_rate=1.)
        loss_functions_rr['hint'] = HINTLoss(
            normalization_rate=1., regularization_rate=1.)
        loss_functions_rr['hint_ig'] = HINTLoss_IG(
            normalization_rate=1., regularization_rate=1.)
        loss_functions_rr['rbr'] = RBRLoss(
            normalization_rate=1., regularization_rate=1.)

        loss_sum_ra_ce = 0.
        loss_sum_ra_non_ce = 0.
        loss_sums_rr = {k: 0. for k in loss_function_keys}

        logging.info('calculating normalization rates ...')

        # iterate all batches of train
        for X, y, E_pnlt, E_rwrd, ce_mask in train_loader:
            X.requires_grad_()

            # compute right-answer loss on CEs
            X_ce, y_ce, _, _ = X[ce_mask], y[ce_mask], E_pnlt[ce_mask], E_rwrd[ce_mask]
            if len(X_ce) > 0:
                y_hat_ce = self.model(X_ce)
                loss_ra_ce = self.loss_function_right_answer(y_hat_ce, y_ce)
                loss_sum_ra_ce += loss_ra_ce

            # compute right-answer loss on non-CEs
            X, y, E_pnlt, E_rwrd = X[~ce_mask], y[~ce_mask], E_pnlt[~ce_mask], E_rwrd[~ce_mask]

            if len(X) > 0:  # required as rrr doesn't work on zero-sized tensors
                y_hat = self.model(X)
                loss_ra_non_ce = self.loss_function_right_answer(y_hat, y)
                loss_sum_ra_non_ce = + loss_ra_non_ce

                # iterate over set of loss function keys to consider
                for k in loss_function_keys:
                    loss_sums_rr[k] += loss_functions_rr[k].forward(
                        self.model, X, y, loss_ra_non_ce, E_rwrd, y_hat, self.device).detach()

        loss_normalization_rates = dict()
        # print(f'loss_sum_ra_ce={loss_sum_ra_ce}, loss_sum_ra_non_ce={loss_sum_ra_non_ce}')

        for k, loss_sum_rr in loss_sums_rr.items():

            # first we want to normalize all rr losses against the entire ra part (ce+non-ce)
            normalized_against_ra = (loss_sum_ra_non_ce) / loss_sum_rr

            # since we use multiple rr losses, we also need devide this value by the number of rr losses
            loss_normalization_rates[k] = normalized_against_ra.detach(
            ) / len(loss_sums_rr)

            # print(f'k={k}, loss_sum_rr={loss_sum_rr}, normalized_loss_sum_rr={loss_sum_rr * loss_normalization_rates[k]}, normalized_against_ra={normalized_against_ra}, normalization_rate={loss_normalization_rates[k]}')

        return loss_normalization_rates

    def fit(
        self,
        train_loader,
        test_loader,
        epochs,
        regularization_rates_rr: dict,

        normalize_loss_functions=True,
        early_stopping_patience=3,
        save_best_epoch=False,
        save_last=True,
    ):
        """
        Fits the learner using training data from dataloader for specified number 
        of epochs. After every epoch the learner is evaluated on the specified
        test_dataloader. Uses pytorch SummaryWriter to log stats for tensorboard.
        Writes the training progress and stats per epoch in a logfile to the logs-folder.

        Args:
            train_data: train data List[(X, y, expl)] where expl are the ground-truth user 
                feedback masks (optional).
            test_dataloader: validation dataloader (Xt, yt).
            epochs: number of epochs to train.
            save_best: saves the best model on the train loss to file.
            save_last: saves the model after every epoch .
        """

        # if lambda None -> don't use loss func
        # if lambda 0. -> evaluate reg-rate for loss func
        # if reg-rate either None nor 0. -> use reg-rate for loss func

        # default normalization rate for losses is 1. (no normalization / neutral element)
        normalization_rates_rr = defaultdict(lambda: 1.)

        if normalize_loss_functions:
            # calculate and overwrite norm. rates for specified loss functions
            # dependant on the selection of loss functions
            normalization_rates_rr.update(
                self.calculate_normalization_rates(
                    train_loader, set(regularization_rates_rr.keys()))
            )
        logging.info(f'normalization_rates={normalization_rates_rr.items()}')
        logging.info(f'regularization_rates={regularization_rates_rr}')

        # instantiate specified loss fucntions (reg rate may be default [1.0] if not further specified on cli arg)
        loss_functions_rr = dict()
        if 'rrr' in regularization_rates_rr:
            loss_functions_rr['rrr'] = RRRLoss(
                normalization_rates_rr['rrr'], regularization_rates_rr['rrr'])
        if 'rrr_gc' in regularization_rates_rr:
            loss_functions_rr['rrr_gc'] = RRRGradCamLoss(
                normalization_rates_rr['rrr_gc'], regularization_rates_rr['rrr_gc'])
        if 'cdep' in regularization_rates_rr:
            loss_functions_rr['cdep'] = CDEPLoss(
                normalization_rates_rr['cdep'], regularization_rates_rr['cdep'])
        if 'hint' in regularization_rates_rr:
            loss_functions_rr['hint'] = HINTLoss(
                normalization_rates_rr['hint'], regularization_rates_rr['hint'])
        if 'hint_ig' in regularization_rates_rr:
            loss_functions_rr['hint_ig'] = HINTLoss_IG(
                normalization_rates_rr['hint_ig'], regularization_rates_rr['hint_ig'])
        if 'rbr' in regularization_rates_rr:
            loss_functions_rr['rbr'] = RBRLoss(
                normalization_rates_rr['rbr'], regularization_rates_rr['rbr'])

        logging.info(f'loss_functions_rr={loss_functions_rr}')

        run_id = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")  # str(uuid.uuid1())

        log_writer = open(f"logs/{self.modelname}_{run_id}.log", "w+")
        log_writer.write(
            "epoch,acc,loss,ra_loss,rr_loss,val_acc,val_loss,time\n")

        tensorboard_writer = SummaryWriter(
            log_dir=f'runs/{self.modelname}_{run_id}')

        logging.info("starting training ...")

        lowest_test_loss = 10_000_000_000  # just a very large number
        elapsed_time = 0
        early_stopping_worse_epochs_counter = 0

        # shift epoch counting by one to start with epoch 1 (instead of 0)
        for epoch in range(self.n_trained_epochs+1, epochs+1):
            # collecting losses of epoch
            epoch_losses = defaultdict(
                lambda: torch.tensor(0., device=self.device))

            self.model.train()
            len_dataset = len(train_loader.dataset)

            # counts the number of correctly classified instances during current epoch
            epoch_correct = 0

            epoch_start_time = time.time()

            # iterate batches
            for X, y, E_pnlt, E_rwrd, ce_mask in tqdm(train_loader):
                logging.debug(
                    f"batch consists of {len(X[~ce_mask])} examples and {len(X[ce_mask])} counterexamples")

                # collecting losses of batch
                batch_losses = defaultdict(
                    lambda: torch.tensor(0., device=self.device))

                self.optimizer.zero_grad()
                X.requires_grad_()

                # compute right-answer loss on CEs
                X_ce, y_ce, _, _ = X[ce_mask], y[ce_mask], E_pnlt[ce_mask], E_rwrd[ce_mask]
                if len(X_ce) > 0:
                    y_hat_ce = self.model(X_ce)
                    epoch_correct += (y_hat_ce.argmax(1) ==
                                      y_ce).type(torch.float).sum().item()
                    loss = self.loss_function_right_answer(y_hat_ce, y_ce)
                    batch_losses['ra_ce'] += loss
                    epoch_losses['ra_ce'] += loss

                # compute right-answer AND right-reason loss on non-CEs
                X, y, E_pnlt, E_rwrd = X[~ce_mask], y[~ce_mask], E_pnlt[~ce_mask], E_rwrd[~ce_mask]
                if len(X) > 0:  # required as rrr doesn't work on zero-sized tensors
                    y_hat = self.model(X)
                    epoch_correct += (y_hat.argmax(1) ==
                                      y).type(torch.float).sum().item()
                    loss = self.loss_function_right_answer(y_hat, y)
                    batch_losses['ra_non_ce'] += loss
                    epoch_losses['ra_non_ce'] += loss

                    # calculate loss functions
                    for k, loss_function in loss_functions_rr.items():
                        loss = loss_function.forward(
                            self.model, X, y, batch_losses['ra_non_ce'], E_rwrd, y_hat, self.device)
                        batch_losses['rr_' + k] += loss
                        epoch_losses['rr_' + k] += loss
                        epoch_losses['rr'] += loss

                # backward over batch losses
                sum(batch_losses.values()).backward()

                self.optimizer.step()

            epoch_loss = epoch_losses['ra_ce'] + \
                epoch_losses['ra_non_ce'] + epoch_losses['rr']

            epoch_correct /= len_dataset
            train_acc = 100. * epoch_correct

            elapsed_time_cur = time.time() - epoch_start_time
            elapsed_time += elapsed_time_cur

            tensorboard_writer.add_scalar('Loss/train', epoch_loss, epoch)

            tensorboard_writer.add_scalar(
                'Loss/right_answer_non_counterexamples', epoch_losses['ra_non_ce'], epoch)
            tensorboard_writer.add_scalar(
                'Loss/right_answer_counterexamples', epoch_losses['ra_ce'], epoch)
            tensorboard_writer.add_scalar(
                'Loss/right_reason_total', epoch_losses['rr'], epoch)

            tensorboard_writer.add_scalar(
                'Loss/right_reason_rrr', epoch_losses['rr_rrr'], epoch)
            tensorboard_writer.add_scalar(
                'Loss/right_reason_rrr_gc', epoch_losses['rr_rrr_gc'], epoch)
            tensorboard_writer.add_scalar(
                'Loss/right_reason_cdep', epoch_losses['rr_cdep'], epoch)
            tensorboard_writer.add_scalar(
                'Loss/right_reason_hint', epoch_losses['rr_hint'], epoch)
            tensorboard_writer.add_scalar(
                'Loss/right_reason_hint_ig', epoch_losses['rr_hint_ig'], epoch)
            tensorboard_writer.add_scalar(
                'Loss/right_reason_rbr', epoch_losses['rr_rbr'], epoch)

            tensorboard_writer.add_scalar('Acc/train', train_acc, epoch)
            tensorboard_writer.add_scalar(
                'Time/train', elapsed_time_cur, epoch)

            test_acc, test_loss = self.score(
                test_loader, self.loss_function_right_answer)

            tensorboard_writer.add_scalar('Loss/test', test_loss, epoch)
            tensorboard_writer.add_scalar('Acc/test', test_acc, epoch)
            tensorboard_writer.flush()

            # todo: find better way to print losses
            losses_summary = ', '.join(map(lambda x: f'{x[0]}={x[1].item():,.3f}' if x[1].item(
            ) != 0. else '-', epoch_losses.items()))
            print(f'E={epoch:0>2} | train_acc={(train_acc):>0.1f}%, train_loss={epoch_loss:>8f} | test_acc={test_acc:>0.1f}%, test_loss={test_loss:>8f} | {losses_summary}')

            # write in logfile -> we need the logfile to see plots in Jupyter notebooks
            log_writer.write(
                f"{epoch},{(train_acc):>0.1f},{epoch_loss:>8f},{epoch_losses['ra_non_ce']:>8f},{epoch_losses['rr']:>8f},{test_acc:>0.1f},{test_loss:>8f},{elapsed_time_cur:>0.4f}\n")
            log_writer.flush()

            # # log to terminal on switch
            # if epoch == disable_xil_loss_first_n_epochs and verbose and disable_xil_loss_first_n_epochs != 0:
            #     bs_store = (epoch, train_acc, train_loss, val_acc, val_loss)

            if save_last:
                self.save_to_checkpoint(n_trained_epochs=epoch)

            # save the current best model on val set
            if test_loss < lowest_test_loss:
                # todo: think about introducing epsilon for thresholing on stagnating loss curve

                lowest_test_loss = test_loss
                early_stopping_worse_epochs_counter = 0

                if save_best_epoch:
                    # set to 50 epochs to prevent resuming training if loaded
                    self.save_to_checkpoint(n_trained_epochs=50, best=True)
            else:
                early_stopping_worse_epochs_counter += 1
                if early_stopping_worse_epochs_counter > early_stopping_patience:

                    # exceeded allowed patience -> stop training
                    logging.info(
                        f'test_loss did not improve within {early_stopping_worse_epochs_counter} epochs')

                    # reset in case we continue with re-calculated normalization-rates
                    early_stopping_worse_epochs_counter = 0

                    if normalize_loss_functions:
                        # instead of breaking now we re-calculate the normalization rates and continue training
                        normalization_rates_rr.update(
                            self.calculate_normalization_rates(
                                train_loader, set(regularization_rates_rr.keys()))
                        )

                    else:
                        break

        log_writer.close()
        tensorboard_writer.close()

        print(f"--> Training took {elapsed_time:>4f} seconds!")

    def score(self, dataloader, criterion, verbose=False):
        """Returns the acc and loss on the specified dataloader."""
        size = len(dataloader.dataset)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data in dataloader:
                X, y = data[0].to(self.device), data[1].to(self.device)
                logits = self.model(X)
                output = F.softmax(logits, dim=1)
                test_loss += criterion(output, y).item()
                correct += (output.argmax(1) ==
                            y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        if verbose:
            print(
                f"Test Error: Acc: {100*correct:>0.1f}%, Avg loss: {test_loss:>8f}")
        return 100*correct, test_loss

    def save_to_checkpoint(self, n_trained_epochs, best=False):
        """
        Save the model dict to disk.
        """

        modelname = self.modelname + '_best' if best else self.modelname

        checkpoint = {
            'weights': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            'modelname': modelname,
            'n_trained_epochs': n_trained_epochs,
            'rng_state': torch.get_rng_state()
        }

        path = f'learner/model_store/{modelname}.pt'
        torch.save(checkpoint, path)
        logging.debug(f'model saved to "{path}"')

    def load_from_checkpoint(self):
        """
        Try to load the model with name from the model_store.
        """
        path = f'learner/model_store/{self.modelname}.pt'

        try:
            checkpoint = torch.load(
                path, map_location=torch.device(self.device))

            self.model.load_state_dict(checkpoint['weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer_dict'])
            torch.set_rng_state(checkpoint['rng_state'].type(torch.ByteTensor))

            try:
                self.n_trained_epochs = checkpoint['n_trained_epochs']
                logging.info(
                    f'Loaded {path}. Was trained for {self.n_trained_epochs} epochs')
            except KeyError:
                # in case we can't find field, just assume it was trained for all (50) epochs
                self.n_trained_epochs = 50

        except FileNotFoundError:
            logging.info(
                f'No checkpoint found for "{path}" -> continue with normal training')
