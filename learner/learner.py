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

        # model automatically tries to load checkpoint from previous run
        self.load_from_checkpoint()


    def evaluate_regularization_rates(self, train_loader):
        """
        Will collect losses of all loss-functions on `train_loader` and return rates to regulate all losses to 1
        """

        loss_sum_ra = 0

        loss_function_rr_rrr = RRRLoss(regularizer_rate=1)
        loss_sum_rr_rrr = 0
        
        loss_function_rrr_gc = RRRGradCamLoss(regularizer_rate=1)
        loss_sum_rrr_gc = 0
        
        loss_function_rr_cdep = CDEPLoss(regularizer_rate=1)
        loss_sum_rr_cdep = 0
        
        loss_function_rr_hint = HINTLoss(regularizer_rate=1)
        loss_sum_rr_hint = 0
        
        loss_function_rr_hint_ig = HINTLoss_IG(regularizer_rate=1)
        loss_sum_rr_hint_ig = 0
        
        loss_function_rr_rbr = RBRLoss(regularizer_rate=1)
        loss_sum_rr_rbr = 0

        # iterate all batches of train
        for X, y, E_pnlt, E_rwrd, ce_mask in tqdm(train_loader, unit="batch"):
            X, y, E_pnlt, E_rwrd = X[~ce_mask], y[~ce_mask], E_pnlt[~ce_mask], E_rwrd[~ce_mask]

            X.requires_grad_()

            if len(X) > 0:  # required as rrr doesn't work on zero-sized tensors
                y_hat = self.model(X)
                loss_ra = self.loss_function_right_answer(y_hat, y)
                loss_sum_ra += loss_ra

                loss_sum_rr_rrr += loss_function_rr_rrr.forward(X, E_rwrd, y_hat)  # todo check changes!
                loss_sum_rrr_gc += loss_function_rrr_gc.forward(self.model, X, y, E_rwrd, y_hat, self.device)
                loss_sum_rr_cdep += loss_function_rr_cdep.forward(self.model, X, y, E_rwrd, self.device)
                loss_sum_rr_hint += loss_function_rr_hint.forward(self.model, X, y, E_rwrd, self.device)  # todo check changes!
                loss_sum_rr_hint_ig += loss_function_rr_hint_ig.forward(self.model, X, E_rwrd, y_hat, self.device)  # todo check changes!
                loss_sum_rr_rbr += loss_function_rr_rbr.forward(self.model, X, y, loss_ra, E_rwrd, y_hat)

        logging.info(f'calculated regularization_rates: rrr={loss_sum_ra/loss_sum_rr_rrr},rrr_gc={loss_sum_ra/loss_sum_rrr_gc},cdep={loss_sum_ra/loss_sum_rr_cdep},hint={loss_sum_ra/loss_sum_rr_hint},hint_ig={loss_sum_ra/loss_sum_rr_hint_ig},rbr={loss_sum_ra/loss_sum_rr_rbr}')
        return \
            (loss_sum_ra/loss_sum_rr_rrr).item(),\
            (loss_sum_ra/loss_sum_rrr_gc).item(),\
            (loss_sum_ra/loss_sum_rr_cdep).item(),\
            (loss_sum_ra/loss_sum_rr_hint).item(),\
            (loss_sum_ra/loss_sum_rr_hint_ig).item(),\
            (loss_sum_ra/loss_sum_rr_rbr).item()

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


    def fit(
            self,
            train_loader,
            test_loader,
            epochs,
            save_best_epoch=False,
            save_last=True,
            loss_rrr_regularizer_rate=None,
            loss_rrr_gc_regularizer_rate=None,
            loss_cdep_regularizer_rate=None,
            loss_hint_regularizer_rate=None,
            loss_hint_ig_regularizer_rate=None,
            loss_rbr_regularizer_rate=None,
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

        run_id = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")  # str(uuid.uuid1())

        log_writer = open(f"logs/{self.modelname}_{run_id}.log", "w+")
        log_writer.write(
            "epoch,acc,loss,ra_loss,rr_loss,val_acc,val_loss,time\n")

        tensorboard_writer = SummaryWriter(
            log_dir=f'runs/{self.modelname}_{run_id}')


        # initialize loss-functions
        # todo: since they are stateless consider refactoring them into single function
        loss_function_rrr = RRRLoss(regularizer_rate=loss_rrr_regularizer_rate) if loss_rrr_regularizer_rate else None
        loss_function_rrr_gc = RRRGradCamLoss(regularizer_rate=loss_rrr_gc_regularizer_rate) if loss_rrr_gc_regularizer_rate else None
        loss_function_cdep = CDEPLoss(regularizer_rate=loss_cdep_regularizer_rate) if loss_cdep_regularizer_rate else None
        loss_function_hint = HINTLoss(regularizer_rate=loss_hint_regularizer_rate) if loss_hint_regularizer_rate else None
        loss_function_hint_ig = HINTLoss_IG(regularizer_rate=loss_hint_ig_regularizer_rate) if loss_hint_ig_regularizer_rate else None
        loss_function_rbr = RBRLoss(regularizer_rate=loss_rbr_regularizer_rate) if loss_rbr_regularizer_rate else None

        print("Start training...")

        best_epoch_loss = 10000000
        elapsed_time = 0

        for epoch in range(self.n_trained_epochs+1, epochs+1):
            # collecting losses of epoch
            epoch_losses = defaultdict(lambda: torch.tensor(0., device=self.device))


            self.model.train()
            len_dataset = len(train_loader.dataset)

            # counts the number of correctly classified instances during current epoch
            epoch_correct = 0

            epoch_start_time = time.time()

            # iterate batches
            for X, y, E_pnlt, E_rwrd, ce_mask in train_loader:
                logging.debug(f"batch consists of {len(X[~ce_mask])} examples and {len(X[ce_mask])} counterexamples")

                # collecting losses of batch
                batch_losses = defaultdict(lambda: torch.tensor(0., device=self.device))

                self.optimizer.zero_grad()
                X.requires_grad_()

                # compute right-answer loss on CEs
                X_ce, y_ce, _, _ = X[ce_mask], y[ce_mask], E_pnlt[ce_mask], E_rwrd[ce_mask]
                if len(X_ce) > 0:
                    y_hat_ce = self.model(X_ce)
                    epoch_correct += (y_hat_ce.argmax(1) == y_ce).type(torch.float).sum().item()
                    loss = self.loss_function_right_answer(y_hat_ce, y_ce)
                    batch_losses['ra_ce'] += loss
                    epoch_losses['ra_ce'] += loss

                # compute right-answer AND right-reason loss on non-CEs
                X, y, E_pnlt, E_rwrd = X[~ce_mask], y[~ce_mask], E_pnlt[~ce_mask], E_rwrd[~ce_mask]
                if len(X) > 0:  # required as rrr doesn't work on zero-sized tensors
                    y_hat = self.model(X)
                    epoch_correct += (y_hat.argmax(1) == y).type(torch.float).sum().item()
                    loss = self.loss_function_right_answer(y_hat, y)
                    batch_losses['ra_non_ce'] += loss
                    epoch_losses['ra_non_ce'] += loss

                    ###################
                    # MultiLoss START #
                    ###################

                    if loss_function_rrr:
                        loss = loss_function_rrr.forward(X, E_rwrd, y_hat)  # todo check changes!
                        
                        batch_losses['rr_rrr'] += loss
                        epoch_losses['rr_rrr'] += loss
                        batch_losses['rr'] += loss
                        epoch_losses['rr'] += loss

                    if loss_function_rrr_gc:
                        loss = loss_function_rrr_gc.forward(self.model, X, y, E_rwrd, y_hat, self.device)
                        
                        batch_losses['rr_rrr_gc'] += loss
                        epoch_losses['rr_rrr_gc'] += loss
                        batch_losses['rr'] += loss
                        epoch_losses['rr'] += loss

                    if loss_function_cdep:
                        loss = loss_function_cdep.forward(self.model, X, y, E_rwrd, self.device)

                        batch_losses['rr_cdep'] += loss
                        epoch_losses['rr_cdep'] += loss
                        batch_losses['rr'] += loss
                        epoch_losses['rr'] += loss

                    if loss_function_hint:
                        loss = loss_function_hint.forward(self.model, X, y, E_rwrd, self.device)  # todo check changes!

                        batch_losses['rr_hint'] += loss
                        epoch_losses['rr_hint'] += loss
                        batch_losses['rr'] += loss
                        epoch_losses['rr'] += loss

                    if loss_function_hint_ig:
                        loss = loss_function_hint_ig.forward(self.model, X, E_rwrd, y_hat, self.device)  # todo check changes!

                        batch_losses['rr_hint_ig'] += loss
                        epoch_losses['rr_hint_ig'] += loss
                        batch_losses['rr'] += loss
                        epoch_losses['rr'] += loss

                    if loss_function_rbr:
                        loss = loss_function_rbr.forward(self.model, X, y, batch_losses['ra_non_ce'], E_rwrd, y_hat)

                        batch_losses['rr_rbr'] += loss
                        epoch_losses['rr_rbr'] += loss
                        batch_losses['rr'] += loss
                        epoch_losses['rr'] += loss

                    #################
                    # MultiLoss END #
                    #################

                (batch_losses['ra_ce'] + batch_losses['ra_non_ce'] + batch_losses['rr']).backward()
                self.optimizer.step()

            epoch_loss = epoch_losses['ra_ce'] + epoch_losses['ra_non_ce'] + epoch_losses['rr']

            epoch_correct /= len_dataset
            train_acc = 100. * epoch_correct

            elapsed_time_cur = time.time() - epoch_start_time
            elapsed_time += elapsed_time_cur

            tensorboard_writer.add_scalar('Loss/train', epoch_loss, epoch)

            tensorboard_writer.add_scalar('Loss/right_answer_non_counterexamples', epoch_losses['ra_non_ce'], epoch)
            tensorboard_writer.add_scalar('Loss/right_answer_counterexamples', epoch_losses['ra_ce'], epoch)
            tensorboard_writer.add_scalar('Loss/right_reason_total', epoch_losses['rr'], epoch)

            tensorboard_writer.add_scalar('Loss/right_reason_rrr', epoch_losses['rr_rrr'], epoch)
            tensorboard_writer.add_scalar('Loss/right_reason_rrr_gc', epoch_losses['rr_rrr_gc'], epoch)
            tensorboard_writer.add_scalar('Loss/right_reason_cdep', epoch_losses['rr_cdep'], epoch)
            tensorboard_writer.add_scalar('Loss/right_reason_hint', epoch_losses['rr_hint'], epoch)
            tensorboard_writer.add_scalar('Loss/right_reason_hint_ig', epoch_losses['rr_hint_ig'], epoch)
            tensorboard_writer.add_scalar('Loss/right_reason_rbr', epoch_losses['rr_rbr'], epoch)

            tensorboard_writer.add_scalar('Acc/train', train_acc, epoch)
            tensorboard_writer.add_scalar('Time/train', elapsed_time_cur, epoch)

            test_acc, test_loss = self.score(test_loader, self.loss_function_right_answer)

            tensorboard_writer.add_scalar('Loss/test', test_loss, epoch)
            tensorboard_writer.add_scalar('Acc/test', test_acc, epoch)
            tensorboard_writer.flush()

            # todo: find better way to print losses
            losses_summary = ', '.join(map(lambda x: f'{x[0]}={x[1].item():,.3f}' if x[1].item() != 0. else '-', epoch_losses.items()))
            print(f'E={epoch:0>2} | train_acc={(train_acc):>0.1f}%, train_loss={epoch_loss:>8f} | test_acc={test_acc:>0.1f}%, test_loss={test_loss:>8f} | {losses_summary}')

            # write in logfile -> we need the logfile to see plots in Jupyter notebooks
            log_writer.write(
                f"{epoch},{(train_acc):>0.1f},{epoch_loss:>8f},{epoch_losses['ra_non_ce']:>8f},{epoch_losses['rr']:>8f},{test_acc:>0.1f},{test_loss:>8f},{elapsed_time_cur:>0.4f}\n")
            log_writer.flush()

            # # log to terminal on switch
            # if epoch == disable_xil_loss_first_n_epochs and verbose and disable_xil_loss_first_n_epochs != 0:
            #     bs_store = (epoch, train_acc, train_loss, val_acc, val_loss)

            # save the current best model on val set
            if save_best_epoch and epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                self.save_to_checkpoint(n_trained_epochs=epoch, best=True)

            if save_last:
                self.save_to_checkpoint(n_trained_epochs=epoch)
            

        log_writer.close()
        tensorboard_writer.close()

        print(f"--> Training took {elapsed_time:>4f} seconds!")


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
            checkpoint = torch.load(path, map_location=torch.device(self.device))
            
            self.model.load_state_dict(checkpoint['weights'])
            self.optimizer.load_state_dict(checkpoint['optimizer_dict'])
            torch.set_rng_state(checkpoint['rng_state'].type(torch.ByteTensor))

            try:
                self.n_trained_epochs = checkpoint['n_trained_epochs']
                print(f'Loaded {path}. Was trained for {self.n_trained_epochs} epochs')
            except KeyError:
                # in case we can't find field, just assume it was trained for all (50) epochs
                self.n_trained_epochs = 50

            
        except FileNotFoundError:
            logging.info(f'No checkpoint found for "{path}" -> continue with normal training')

