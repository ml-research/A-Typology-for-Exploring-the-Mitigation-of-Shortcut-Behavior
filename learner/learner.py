import logging
import time
import uuid
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from rtpt import RTPT
from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, HINTLoss_IG, RBRLoss
from datetime import datetime


class Learner:

    def __init__(self, model, optimizer, device, modelname, base_criterion=F.cross_entropy,
                 loss_rrr_weight=None, loss_weight_rrr_gc=None, loss_weight_cdep=None, loss_weight_hint=None, loss_weight_rbr=None, load=False):

        self.model = model  # handled by save()
        self.optimizer = optimizer  # handled by save()
        self.device = device
        self.modelname = modelname  # handled by save()

        if loss_rrr_weight:
            self.loss_function_rrr = RRRLoss(weight=loss_rrr_weight)

        if torch.is_tensor(loss_weight_rrr_gc):
            self.loss_function_rrr_gc = RRRGradCamLoss(
                weight=loss_weight_rrr_gc)

        if torch.is_tensor(loss_weight_cdep):
            self.loss_function_cdep = CDEPLoss(weight=loss_weight_cdep)

        if torch.is_tensor(loss_weight_hint):
            self.loss_function_hint = HINTLoss(weight=loss_weight_hint)

        if torch.is_tensor(loss_weight_rbr):
            self.loss_function_rbr = RBRLoss(weight=loss_weight_rbr)

        self.base_criterion = base_criterion

        run_id = datetime.now().strftime("%d-%m-%Y_%H:%M:%S") # str(uuid.uuid1())
        
        self.log_writer = open(f"logs/{self.modelname}_{run_id}.log", "w+")
        self.tensorboard_writer = SummaryWriter(
            log_dir=f"runs/{self.modelname}_{run_id}", comment=f"{self.modelname}_{run_id}")
        
        if load:
            self.load(modelname+'.pt')

    def load(self, name):
        # TODO: adapt to class implementation; may have introduced breaking changes
        """Load the model with name from the model_store."""
        checkpoint = torch.load(
            'learner/model_store/' + name, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['weights'])
        epochs_ = "none"
        if 'optimizer_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_dict'])
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].type(torch.ByteTensor))
        if 'epochs' in checkpoint:
            epochs_ = checkpoint['epochs']
        print(
            f"Model {self.modelname} loaded! Was trained on {checkpoint['loss']} for {epochs_} epochs!")

    def save_learner(self, epochs=0, verbose=1, best=False):
        pass
        """Save the model dict to disk."""
        # # TODO: adapt to class implementation; may have introduced breaking changes
        # if best:
        #    self.modelname = self.modelname + "-bestOnTrain"
        results = {
            'weights': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            'modelname': self.modelname,
            'epochs': epochs,
            'rng_state':  torch.get_rng_state()
        }

        if self.loss_function_rrr:
            results['loss_rrr'] = str(self.loss_function_rrr)

        if self.loss_function_rrr_gc:
            results['loss_rrr_gc'] = str(self.loss_function_rrr_gc)

        if self.loss_function_cdep:
            results['loss_cdep'] = str(self.loss_function_cdep)

        if self.loss_function_hint:
            results['loss_hint'] = str(self.loss_function_hint)

        if self.loss_function_rbr:
            results['loss_rbr'] = str(self.loss_function_rbr)

        torch.save(results, 'learner/model_store/' + self.modelname + '.pt')
        if verbose == 1:
            print("Model saved!")

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

    def fit(self, train_loader, test_loader, epochs, save_best_epoch=False, save_last=True,
            verbose=True, verbose_after_n_epochs=1):
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
            verbose: set to False if you want no print outputs.
            verbose_after_n_epochs: print outputs after every n epochs. 
        """

        print("Start training...")

        self.log_writer.write("epoch,acc,loss,ra_loss,rr_loss,val_acc,val_loss,time\n")

        best_epoch_loss = 100000
        elapsed_time = 0

        for epoch in range(1, epochs+1):
            self.model.train()
            len_dataset = len(train_loader.dataset)

            # sums of losses within epoch
            epoch_loss_right_answer_ce = 0.
            epoch_loss_right_answer = 0.
            epoch_loss_right_reason = 0.
            epoch_loss_hint = 0.
            epoch_loss_rrr = 0.
            epoch_loss_rrr_gc = 0.
            epoch_loss_cdep = 0.
            epoch_loss_rbr = 0.

            epoch_correct = 0

            epoch_start_time = time.time()

            for X, y, E_pnlt, E_rwrd, ce_mask in tqdm(train_loader, unit='batch'):

                self.optimizer.zero_grad()
                X.requires_grad_()

                logging.info(f"batch consists of {len(X[~ce_mask])} examples and {len(X[ce_mask])} counterexamples")

                # initialize zero-loss tensors (as they may be unbound)
                loss_right_answer_ce = torch.tensor(0., device=self.device)
                loss_right_answer = torch.tensor(0., device=self.device)
                loss_right_reason = torch.tensor(0., device=self.device)

                # compute right-answer loss on CEs
                X_ce, y_ce, _, _ = X[ce_mask], y[ce_mask], E_pnlt[ce_mask], E_rwrd[ce_mask]
                if len(X_ce) > 0:
                    y_hat_ce = self.model(X_ce)
                    epoch_correct += (y_hat_ce.argmax(1) ==
                                        y_ce).type(torch.float).sum().item()
                    loss_right_answer_ce += self.base_criterion(
                        y_hat_ce, y_ce)
                    epoch_loss_right_answer_ce += loss_right_answer_ce
                    logging.info(
                        f"loss_right_answer_ce={loss_right_answer_ce}")

                # compute right-answer AND right-reason loss on non-CEs
                X, y, E_pnlt, E_rwrd = X[~ce_mask], y[~ce_mask], E_pnlt[~ce_mask], E_rwrd[~ce_mask]
                if len(X) > 0:  # required as rrr doesn't work on zero-sized tensors
                    y_hat = self.model(X)
                    epoch_correct += (y_hat.argmax(1) ==
                                        y).type(torch.float).sum().item()
                    loss_right_answer = self.base_criterion(y_hat, y)
                    epoch_loss_right_answer += loss_right_answer
                    logging.info(f"loss_right_answer={loss_right_answer}")

                    ###################
                    # MultiLoss START #
                    ###################

                    if self.loss_function_rrr:
                        loss_rrr = self.loss_function_rrr.forward(
                            X, y, E_rwrd, y_hat)  # todo check implementation changes!
                        # print(f"loss_rrr={loss_rrr}")
                        loss_right_reason += loss_rrr
                        epoch_loss_rrr += loss_rrr

                    if self.loss_function_rrr_gc:
                        loss_rrr_gc = self.loss_function_rrr_gc.forward(
                            self.model, X, y, E_rwrd, y_hat, self.device)  # changes verified
                        # print(f"loss_rrr_gc={loss_rrr_gc}")
                        loss_right_reason += loss_rrr_gc
                        epoch_loss_rrr_gc += loss_rrr_gc

                    if self.loss_function_cdep:
                        loss_cdep = self.loss_function_cdep.forward(
                            self.model, X, y, E_rwrd, self.device)  # changes verified
                        # print(f"loss_cdep={loss_cdep}")
                        loss_right_reason += loss_cdep
                        epoch_loss_cdep += loss_cdep

                    if self.loss_function_hint:
                        loss_hint = self.loss_function_hint.forward(
                            self.model, X, y, E_rwrd, self.device)  # todo check implementation changes!
                        # print(f"loss_hint={loss_hint}")
                        loss_right_reason += loss_hint
                        epoch_loss_hint += loss_hint

                    if self.loss_function_rbr:
                        loss_rbr = self.loss_function_rbr.forward(
                            self.model, X, y, E_rwrd, y_hat)  # changes verified
                        # print(f"loss_rbr={loss_rbr}")
                        loss_right_reason += loss_rbr
                        epoch_loss_rbr += loss_rbr

                    epoch_loss_right_reason += loss_right_reason
                    logging.info(f"loss_right_reason={loss_right_reason}")

                    #################
                    # MultiLoss END #
                    #################


                (loss_right_answer_ce + loss_right_answer +
                    loss_right_reason).backward()
                self.optimizer.step()

            # print(f"losses after epoch: right_answer={epoch_loss_right_answer}, hint={epoch_loss_hint}, rrr={epoch_loss_rrr}, rrr_gc={epoch_loss_rrr_gc}, cdep={epoch_loss_cdep}, rbr={epoch_loss_rbr}")

            epoch_loss_right_answer /= len_dataset
            epoch_loss_right_reason /= len_dataset
            epoch_correct /= len_dataset
            epoch_loss = epoch_loss_right_answer + epoch_loss_right_reason
            train_acc = 100. * epoch_correct

            elapsed_time_cur = time.time() - epoch_start_time
            elapsed_time += elapsed_time_cur

            val_acc, val_loss = self.score(
                test_loader, self.base_criterion)

            self.tensorboard_writer.add_scalar('Loss/train', epoch_loss, epoch)

            self.tensorboard_writer.add_scalar(
                'Loss/ra', epoch_loss_right_answer, epoch)
            self.tensorboard_writer.add_scalar(
                'Loss/rr', epoch_loss_right_reason, epoch)

            self.tensorboard_writer.add_scalar(
                'Loss/hint', epoch_loss_hint, epoch)
            self.tensorboard_writer.add_scalar(
                'Loss/rrr', epoch_loss_rrr, epoch)
            self.tensorboard_writer.add_scalar(
                'Loss/rrr_gc', epoch_loss_rrr_gc, epoch)
            self.tensorboard_writer.add_scalar(
                'Loss/cdep', epoch_loss_cdep, epoch)
            self.tensorboard_writer.add_scalar(
                'Loss/rbr', epoch_loss_rbr, epoch)
            # self.writer.add_scalar('Loss/ce', epoch_loss_ce, epoch)

            self.tensorboard_writer.add_scalar('Acc/train', train_acc, epoch)
            self.tensorboard_writer.add_scalar(
                'Time/train', elapsed_time_cur, epoch)

            self.tensorboard_writer.add_scalar('Loss/test', val_loss, epoch)
            self.tensorboard_writer.add_scalar('Acc/test', val_acc, epoch)

            self.tensorboard_writer.flush()

            logging.info(
                f"Epoch {epoch}| accuracy: {(train_acc):>0.1f}%, loss: {epoch_loss:>8f} | Test Error: Acc: {val_acc:>0.1f}%, Avg loss: {val_loss:>8f}")

            # test acc on test set

            # write in logfile -> we need the logfile to see plots in Jupyter notebooks
            self.log_writer.write(f"{epoch},{(train_acc):>0.1f},{epoch_loss:>8f},{epoch_loss_right_answer:>8f},{epoch_loss_right_reason:>8f},{(val_acc):>0.1f},{val_loss:>8f},{elapsed_time_cur:>0.4f}\n")
            self.log_writer.flush()

            # # log to terminal on switch
            # if epoch == disable_xil_loss_first_n_epochs and verbose and disable_xil_loss_first_n_epochs != 0:
            #     bs_store = (epoch, train_acc, train_loss, val_acc, val_loss)

            # save the current best model on val set
            if save_best_epoch and epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                self.save_learner(verbose=False, best=True)

            if save_last:
                self.save_learner(verbose=False)

        print(f"--> Training took {elapsed_time:>4f} seconds!")
        self.tensorboard_writer.close()

        # # return the train_acc and loss after last epoch
        # return train_acc, train_loss, elapsed_time, bs_store
