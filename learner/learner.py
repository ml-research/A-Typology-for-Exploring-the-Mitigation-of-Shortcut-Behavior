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
            log_dir=f"runs/{self.modelname}_{run_id}", comment=f"{self.modelname}_{run_id}")


        # initialize loss-functions
        # todo: since they are stateless consider refactoring them into single function
        loss_function_rrr = RRRLoss(regularizer_rate=loss_rrr_regularizer_rate) if loss_rrr_regularizer_rate else None
        loss_function_rrr_gc = RRRGradCamLoss(regularizer_rate=loss_rrr_gc_regularizer_rate) if loss_rrr_gc_regularizer_rate else None
        loss_function_cdep = CDEPLoss(regularizer_rate=loss_cdep_regularizer_rate) if loss_cdep_regularizer_rate else None
        loss_function_hint = HINTLoss(regularizer_rate=loss_hint_regularizer_rate) if loss_hint_regularizer_rate else None
        loss_function_hint_ig = HINTLoss_IG(regularizer_rate=loss_hint_ig_regularizer_rate) if loss_hint_ig_regularizer_rate else None
        loss_function_rbr = RBRLoss(regularizer_rate=loss_rbr_regularizer_rate) if loss_rbr_regularizer_rate else None

        print("Start training...")

        best_epoch_loss = 100000
        elapsed_time = 0

        for epoch in range(self.n_trained_epochs+1, epochs+1):
            self.model.train()
            len_dataset = len(train_loader.dataset)

            # sums of losses within epoch
            epoch_loss_right_answer_ce = 0.
            epoch_loss_right_answer = 0.
            epoch_loss_right_reason = 0.
            epoch_loss_rrr = 0.
            epoch_loss_rrr_gc = 0.
            epoch_loss_cdep = 0.
            epoch_loss_hint = 0.
            epoch_loss_hint_ig = 0.
            epoch_loss_rbr = 0.

            epoch_correct = 0

            epoch_start_time = time.time()

            for X, y, E_pnlt, E_rwrd, non_ce_mask in tqdm(train_loader, unit='batch'):

                self.optimizer.zero_grad()
                X.requires_grad_()

                logging.debug(f"batch consists of {len(X[non_ce_mask])} examples and {len(X[~non_ce_mask])} counterexamples")

                # initialize zero-loss tensors (as they may be unbound)
                batch_loss_right_answer_ce = torch.tensor(
                    0., device=self.device)
                batch_loss_right_answer = torch.tensor(0., device=self.device)
                batch_loss_right_reason = torch.tensor(0., device=self.device)

                # compute right-answer loss on CEs
                X_ce, y_ce, _, _ = X[~non_ce_mask], y[~non_ce_mask], E_pnlt[~non_ce_mask], E_rwrd[~non_ce_mask]
                if len(X_ce) > 0:
                    y_hat_ce = self.model(X_ce)
                    epoch_correct += (y_hat_ce.argmax(1) ==
                                      y_ce).type(torch.float).sum().item()
                    batch_loss_right_answer_ce += self.loss_function_right_answer(
                        y_hat_ce, y_ce)
                    epoch_loss_right_answer_ce += batch_loss_right_answer_ce
                    logging.debug(
                        f"loss_right_answer_ce={batch_loss_right_answer_ce}")

                # compute right-answer AND right-reason loss on non-CEs
                X, y, E_pnlt, E_rwrd = X[non_ce_mask], y[non_ce_mask], E_pnlt[non_ce_mask], E_rwrd[non_ce_mask]
                if len(X) > 0:  # required as rrr doesn't work on zero-sized tensors
                    y_hat = self.model(X)
                    epoch_correct += (y_hat.argmax(1) ==
                                      y).type(torch.float).sum().item()
                    batch_loss_right_answer = self.loss_function_right_answer(y_hat, y)
                    epoch_loss_right_answer += batch_loss_right_answer
                    logging.debug(
                        f"loss_right_answer={batch_loss_right_answer}")

                    ###################
                    # MultiLoss START #
                    ###################

                    if loss_function_rrr:
                        batch_loss_rrr = loss_function_rrr.forward(
                            X, E_rwrd, y_hat)  # todo check implementation changes!
                        # print(f"loss_rrr={loss_rrr}")
                        batch_loss_right_reason += batch_loss_rrr
                        epoch_loss_rrr += batch_loss_rrr

                    if loss_function_rrr_gc:
                        batch_loss_rrr_gc = loss_function_rrr_gc.forward(
                            self.model, X, y, E_rwrd, y_hat, self.device)  # changes verified
                        # print(f"loss_rrr_gc={loss_rrr_gc}")
                        batch_loss_right_reason += batch_loss_rrr_gc
                        epoch_loss_rrr_gc += batch_loss_rrr_gc

                    if loss_function_cdep:
                        batch_loss_cdep = loss_function_cdep.forward(
                            self.model, X, y, E_rwrd, self.device)  # changes verified
                        # print(f"loss_cdep={loss_cdep}")
                        batch_loss_right_reason += batch_loss_cdep
                        epoch_loss_cdep += batch_loss_cdep

                    if loss_function_hint:
                        batch_loss_hint = loss_function_hint.forward(
                            self.model, X, y, E_rwrd, self.device)  # todo check implementation changes!
                        # print(f"loss_hint={loss_hint}")
                        batch_loss_right_reason += batch_loss_hint
                        epoch_loss_hint += batch_loss_hint

                    if loss_function_hint_ig:
                        batch_loss_hint_ig = loss_function_hint_ig.forward(
                            self.model, X, E_rwrd, y_hat, self.device)  # todo check implementation changes!
                        # print(f"loss_hint={batch_loss_hint_ig}")
                        batch_loss_right_reason += batch_loss_hint_ig
                        epoch_loss_hint_ig += batch_loss_hint_ig

                    if loss_function_rbr:
                        batch_loss_rbr = loss_function_rbr.forward(
                            self.model, X, y, batch_loss_right_answer, E_rwrd, y_hat)  # changes verified
                        # print(f"loss_rbr={loss_rbr}")
                        batch_loss_right_reason += batch_loss_rbr
                        epoch_loss_rbr += batch_loss_rbr

                    epoch_loss_right_reason += batch_loss_right_reason
                    logging.debug(
                        f"loss_right_reason={batch_loss_right_reason}")

                    #################
                    # MultiLoss END #
                    #################

                (batch_loss_right_answer_ce + batch_loss_right_answer +
                    batch_loss_right_reason).backward()
                self.optimizer.step()

            print(f"epoch losses: right_answer={epoch_loss_right_answer}, hint={epoch_loss_hint}, hint_ig={epoch_loss_hint_ig}, rrr={epoch_loss_rrr}, rrr_gc={epoch_loss_rrr_gc}, cdep={epoch_loss_cdep}, rbr={epoch_loss_rbr}")

            epoch_loss_right_answer /= len_dataset
            epoch_loss_right_reason /= len_dataset
            epoch_correct /= len_dataset
            train_acc = 100. * epoch_correct
            epoch_loss = epoch_loss_right_answer_ce + \
                epoch_loss_right_answer + epoch_loss_right_reason

            elapsed_time_cur = time.time() - epoch_start_time
            elapsed_time += elapsed_time_cur

            tensorboard_writer.add_scalar('Loss/train', epoch_loss, epoch)

            tensorboard_writer.add_scalar(
                'Loss/ra', epoch_loss_right_answer, epoch)
            tensorboard_writer.add_scalar(
                'Loss/rr', epoch_loss_right_reason, epoch)

            tensorboard_writer.add_scalar(
                'Loss/rrr', epoch_loss_rrr, epoch)
            tensorboard_writer.add_scalar(
                'Loss/rrr_gc', epoch_loss_rrr_gc, epoch)
            tensorboard_writer.add_scalar(
                'Loss/cdep', epoch_loss_cdep, epoch)
            tensorboard_writer.add_scalar(
                'Loss/hint', epoch_loss_hint, epoch)
            tensorboard_writer.add_scalar(
                'Loss/hint_ig', epoch_loss_hint_ig, epoch)
            tensorboard_writer.add_scalar(
                'Loss/rbr', epoch_loss_rbr, epoch)
            tensorboard_writer.add_scalar(
                'Loss/ce', epoch_loss_right_answer_ce, epoch)

            tensorboard_writer.add_scalar('Acc/train', train_acc, epoch)
            tensorboard_writer.add_scalar(
                'Time/train', elapsed_time_cur, epoch)

            val_acc, val_loss = self.score(
                test_loader, self.loss_function_right_answer)

            tensorboard_writer.add_scalar('Loss/test', val_loss, epoch)
            tensorboard_writer.add_scalar('Acc/test', val_acc, epoch)
            tensorboard_writer.flush()

            logging.info(
                f"Epoch {epoch}| accuracy: {(train_acc):>0.1f}%, loss: {epoch_loss:>8f} | Test Error: Acc: {val_acc:>0.1f}%, Avg loss: {val_loss:>8f}")

            # write in logfile -> we need the logfile to see plots in Jupyter notebooks
            log_writer.write(
                f"{epoch},{(train_acc):>0.1f},{epoch_loss:>8f},{epoch_loss_right_answer:>8f},{epoch_loss_right_reason:>8f},{(val_acc):>0.1f},{val_loss:>8f},{elapsed_time_cur:>0.4f}\n")
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

        checkpoint = {
            'weights': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            'modelname':  self.modelname + "-bestOnTrain" if best else self.modelname,
            'n_trained_epochs': n_trained_epochs,
            'rng_state': torch.get_rng_state()
        }

        path = f'learner/model_store/{self.modelname}.pt'
        torch.save(checkpoint, path)
        logging.info(f'model saved to "{path}"')


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
            self.n_trained_epochs = checkpoint['n_trained_epochs']

            print(f'Loaded {path}. Was trained for {self.n_trained_epochs} epochs')
            
        except FileNotFoundError:
            logging.info(f'No checkpoint found for "{path}" -> continue with normal training')

