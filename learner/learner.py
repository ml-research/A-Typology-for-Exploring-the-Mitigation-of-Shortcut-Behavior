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


class Learner:

    def __init__(self, model, optimizer, device, modelname,
                 loss_rrr_weight=None, loss_rrr_gc_weight=None, loss_cdep_weight=None, loss_hint_weight=None, loss_rbr_weight=None, loss_ce=False,
                 base_criterion=F.cross_entropy, load=False):

        self.model = model  # handled by save()
        self.optimizer = optimizer  # handled by save()
        self.device = device
        self.modelname = modelname  # handled by save()

        if loss_rrr_weight:
            self.loss_function_rrr = RRRLoss(weight=loss_rrr_weight)

        if torch.is_tensor(loss_rrr_gc_weight):
            self.loss_function_rrr_gc = RRRGradCamLoss(
                weight=loss_rrr_gc_weight)

        if torch.is_tensor(loss_cdep_weight):
            self.loss_function_cdep = CDEPLoss(weight=loss_cdep_weight)

        if torch.is_tensor(loss_hint_weight):
            self.loss_function_hint = HINTLoss(weight=loss_hint_weight)

        if torch.is_tensor(loss_rbr_weight):
            self.loss_function_rbr = RBRLoss(weight=loss_rbr_weight)

        self.loss_ce = loss_ce
        self.base_criterion = base_criterion

        self.run_id = str(uuid.uuid1())
        self.log_folder = 'logs/' + self.modelname
        self.writer = SummaryWriter(log_dir='runs/' + self.modelname + "--" + self.run_id, comment="_" +
                                    self.modelname + "_id_{}".format(self.run_id))
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

        # todo: add ce loss

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
            verbose=True, verbose_after_n_epochs=1, disable_xil_loss_first_n_epochs=0):
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
            disable_xil_loss_first_n_epochs: disables the XIL loss for the first n epochs 
                falling back to the self.base_criterion. This is used to switch XIL on after 
                n-epochs. If 0 then XIL is used for the whole training.  
        """

        print("Start training...")
        with open(f"{self.log_folder}.log", "w+") as f:
            f.write("epoch,acc,loss,ra_loss,rr_loss,val_acc,val_loss,time\n")
            best_epoch_loss = 100000
            elapsed_time = 0

            for epoch in range(1, epochs+1):
                self.model.train()
                len_dataset = len(train_loader.dataset)

                # sums of losses within epoch
                epoch_loss_right_answer = 0.
                epoch_loss_right_reason = 0.
                epoch_loss_hint = 0.
                epoch_loss_rrr = 0.
                epoch_loss_rrr_gc = 0.
                epoch_loss_cdep = 0.
                epoch_loss_rbr = 0.
                epoch_loss_ce = 0.

                epoch_correct = 0

                epoch_start_time = time.time()

                for batch_index, (X, y, E_pnlt, E_rwrd, counterexample_mask) in enumerate(tqdm(train_loader, unit='batch')):
                    self.optimizer.zero_grad()
                    X.requires_grad_()

                    X_ce = X[counterexample_mask]
                    # y_ce = y[counterexample_mask]
                    # E_pnlt_ce = E_pnlt[counterexample_mask]
                    # E_rwrd_ce = E_rwrd[counterexample_mask]

                    X = X[~counterexample_mask]
                    y = y[~counterexample_mask]
                    E_pnlt = E_pnlt[~counterexample_mask]
                    E_rwrd = E_rwrd[~counterexample_mask]

                    # print(f"X_ce={len(X_ce)}, y_ce={len(y_ce)}, E_pnlt_ce={len(E_pnlt_ce)}, E_rwrd_ce={len(E_rwrd_ce)}\nX_nce={len(X)}, y_nce={len(y)}, E_pnlt_nce={len(E_pnlt)}, E_rwrd_nce={len(E_rwrd)}")

                    y_hat = self.model(X)
                    epoch_correct += (y_hat.argmax(1) ==
                                      y).type(torch.float).sum().item()
                    loss_right_answer = self.base_criterion(y_hat, y)
                    epoch_loss_right_answer += loss_right_answer
                    loss = loss_right_answer

                    # initialize zero loss tensors
                    loss_right_reason = torch.tensor(0.)

                    # functionality to disable xil_loss for n number of epochs beginning from first epoch
                    if epoch <= disable_xil_loss_first_n_epochs:
                        # continue with next batch
                        continue

                    ###################
                    # MultiLoss START #
                    ###################

                    # log to terminal on switch
                    if epoch == disable_xil_loss_first_n_epochs+1 and batch_index == 0 \
                            and verbose and disable_xil_loss_first_n_epochs != 0:
                        logging.info(
                            f"--> XIL loss activated in epoch {epoch}!")

                    logging.info(
                        f"batch consists of {len(X)} examples and {len(X_ce)} counterexamples")
                    # print(f"E_rwrd={E_rwrd[100]}")

                    if self.loss_function_rrr:
                        loss_rrr = self.loss_function_rrr.forward(  # todo check implementation changes!
                            X, y, E_rwrd, y_hat)
                        # print(f"loss_rrr={loss_rrr}")
                        loss_right_reason += loss_rrr
                        epoch_loss_rrr += loss_rrr
                        loss += loss_rrr

                    if self.loss_function_rrr_gc:
                        loss_rrr_gc = self.loss_function_rrr_gc.forward(
                            self.model, X, y, E_rwrd, y_hat, self.device)  # changes verified
                        # print(f"loss_rrr_gc={loss_rrr_gc}")
                        loss_right_reason += loss_rrr_gc
                        epoch_loss_rrr_gc += loss_rrr_gc
                        loss += loss_rrr_gc

                    if self.loss_function_cdep:
                        loss_cdep = self.loss_function_cdep.forward(
                            self.model, X, y, E_rwrd, self.device)  # changes verified
                        # print(f"loss_cdep={loss_cdep}")
                        loss_right_reason += loss_cdep
                        epoch_loss_cdep += loss_cdep
                        loss += loss_cdep

                    if self.loss_function_hint:
                        loss_hint = self.loss_function_hint.forward(
                            self.model, X, y, E_rwrd, self.device)  # todo check implementation changes!
                        # print(f"loss_hint={loss_hint}")
                        loss_right_reason += loss_hint
                        epoch_loss_hint += loss_hint
                        loss += loss_hint

                    if self.loss_function_rbr:
                        loss_rbr = self.loss_function_rbr.forward(
                            self.model, X, y, E_rwrd, y_hat)  # changes verified
                        # print(f"loss_rbr={loss_rbr}")
                        loss_right_reason += loss_rbr
                        epoch_loss_rbr += loss_rbr
                        loss += loss_rbr

                    # TODO
                    # if self.loss_ce:
                    #     self.loss_ce(X_ce)

                    epoch_loss_right_reason += loss_right_reason

                    #################
                    # MultiLoss END #
                    #################

                    loss.backward()
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

                self.writer.add_scalar('Loss/train', epoch_loss, epoch)

                self.writer.add_scalar(
                    'Loss/ra', epoch_loss_right_answer, epoch)
                self.writer.add_scalar(
                    'Loss/rr', epoch_loss_right_reason, epoch)

                self.writer.add_scalar('Loss/hint', epoch_loss_hint, epoch)
                self.writer.add_scalar('Loss/rrr', epoch_loss_rrr, epoch)
                self.writer.add_scalar('Loss/rrr_gc', epoch_loss_rrr_gc, epoch)
                self.writer.add_scalar('Loss/cdep', epoch_loss_cdep, epoch)
                self.writer.add_scalar('Loss/rbr', epoch_loss_rbr, epoch)
                # self.writer.add_scalar('Loss/ce', epoch_loss_ce, epoch)

                self.writer.add_scalar('Acc/train', train_acc, epoch)
                self.writer.add_scalar('Time/train', elapsed_time_cur, epoch)

                self.writer.add_scalar('Loss/test', val_loss, epoch)
                self.writer.add_scalar('Acc/test', val_acc, epoch)

                self.writer.flush()

                # printing acc and loss
                if verbose and (epoch % verbose_after_n_epochs == 0):
                    print(f"Epoch {epoch}| ", end='')
                    print(
                        f"accuracy: {(train_acc):>0.1f}%, loss: {epoch_loss:>8f} | ", end='')
                    print(
                        f"Test Error: Acc: {val_acc:>0.1f}%, Avg loss: {val_loss:>8f}")
                    # print(f" --number RAWR [{how_many_rawr_epoch}]")

                # test acc on test set

                # write in logfile -> we need the logfile to see plots in Jupyter notebooks
                f.write(f"{epoch},{(train_acc):>0.1f},{epoch_loss:>8f},{epoch_loss_right_answer:>8f},{epoch_loss_right_reason:>8f},{(val_acc):>0.1f},{val_loss:>8f},{elapsed_time_cur:>0.4f}\n")

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
        self.writer.close()

        # # return the train_acc and loss after last epoch
        # return train_acc, train_loss, elapsed_time, bs_store
