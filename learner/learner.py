"""Learner class with different utility functions."""
import logging
import time
import uuid
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from rtpt import RTPT

from xil_methods.xil_loss import RRRGradCamLoss, RRRLoss, CDEPLoss, HINTLoss, RBRLoss

class Learner:
    """Implements a ML learner (based on PyTorch model)."""

    def __init__(self, model, loss, optimizer, device, modelname, load=False, \
        base_criterion=F.cross_entropy):
        """
        Args:
            model: pytorch model.
            loss: pytorch loss function.
            optimizer: pytorch optimizer.
            device: either 'cuda' or 'cpu'.
            modelname: a unique name that identifies the model. Is used to store log/run files
                as well as the model itself.
            load: if True loads the model with modelname from model_store.
            base_criterion: pytorch functional loss function. Is used in experiments which 
                disbale the XIL loss. Default cross_entropy.
        """
        self.model = model
        self.loss = loss
        self.base_criterion = base_criterion
        self.optimizer = optimizer
        self.device = device
        self.modelname = modelname
        self.run_id = str(uuid.uuid1())
        self.log_folder = 'logs/' + self.modelname
        self.writer = SummaryWriter(log_dir='runs/' + self.modelname + "--" + self.run_id, comment="_" + \
            self.modelname + "_id_{}".format(self.run_id))
        if load:
            self.load(modelname+'.pt')

    def fit(self, dataloader, test_dataloader, epochs, save_best=False, save_last=True, \
        verbose=True, verbose_after_n_epochs=1, disable_xil_loss_first_n_epochs=0):
        """
        Fits the learner using training data from dataloader for specified number 
        of epochs. After every epoch the learner is evaluated on the specified
        test_dataloader. Uses pytorch SummaryWriter to log stats for tensorboard.
        Writes the training progress and stats per epoch in a logfile to the logs-folder.

        Args:
            dataloader: train dataloader (X, y, expl) where expl are the ground-truth user 
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
        bs_store = None
        with open(f"{self.log_folder}.log", "w") as f:
            f.write("epoch,acc,loss,ra_loss,rr_loss,val_acc,val_loss,time\n")
            best_train_loss = 100000
            elapsed_time = 0

            for epoch in range(1, epochs+1):
                #how_many_rawr_epoch = torch.tensor([0])

                self.model.train()
                size = len(dataloader.dataset)
                # ra_loss = right answer, rr_loss = right reason 
                train_loss, correct, ra_loss, rr_loss = 0, 0, 0, 0
                start_time = time.time()
                
                for batch, data in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    # functionality to disable xil_loss for n number 
                    # of epochs beginning from first epoch
                    if epoch > disable_xil_loss_first_n_epochs:

                        # log to terminal on switch
                        if epoch == disable_xil_loss_first_n_epochs+1 and batch==0 \
                            and verbose and disable_xil_loss_first_n_epochs != 0:
                            print(f"--> XIL loss activated in epoch {epoch}!")


                        if isinstance(self.loss, RRRLoss):
                            X, y, expl = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                            X.requires_grad_()
                            output = self.model(X)
                            loss, ra_loss_c, rr_loss_c = self.loss(X, y, expl, output)
                        
                        elif isinstance(self.loss, RRRGradCamLoss):
                            X, y, expl = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                            X.requires_grad_()
                            output = self.model(X)
                            expl = expl.float()
                            loss, ra_loss_c, rr_loss_c = self.loss(self.model, X, y, expl, output, self.device)
                        
                        elif isinstance(self.loss, CDEPLoss):
                            X, y, expl = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                            output = self.model(X)
                            expl = expl.float()
                            loss, ra_loss_c, rr_loss_c = self.loss(self.model, X, y, expl, output, self.device)

                        elif isinstance(self.loss, HINTLoss):
                            X, y, expl = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                            X.requires_grad_()
                            output = self.model(X)
                            expl = expl.float()
                            loss, ra_loss_c, rr_loss_c = self.loss(self.model, X, y, expl, output, self.device)
                        
                        elif isinstance(self.loss, RBRLoss):
                            X, y, expl = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                            X.requires_grad_()
                            output = self.model(X)
                            expl = expl.float()
                            loss, ra_loss_c, rr_loss_c = self.loss(self.model, X, y, expl, output)

                        else:
                            X, y = data[0].to(self.device), data[1].to(self.device)
                            output = self.model(X)
                            loss = self.loss(output, y)

                    else:
                        X, y = data[0].to(self.device), data[1].to(self.device)
                        output = self.model(X)
                        loss = self.base_criterion(output, y)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()
                    
                    # for calculating loss, acc per epoch
                    train_loss += loss.item()
                    correct += (output.argmax(1) == y).type(torch.float).sum().item()

                    # for tracking right answer and right reason loss
                    if isinstance(self.loss, (RRRLoss, HINTLoss, CDEPLoss, RBRLoss, RRRGradCamLoss)) \
                        and (epoch) > disable_xil_loss_first_n_epochs:
                        ra_loss += ra_loss_c.item()
                        rr_loss += rr_loss_c.item() 
                
                train_loss /= size
                correct /= size
                ra_loss /= size
                rr_loss /= size
                
                train_acc = 100.*correct
                
                end_time = time.time()
                elapsed_time_cur = end_time - start_time
                elapsed_time += elapsed_time_cur

                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/ra', ra_loss, epoch)
                self.writer.add_scalar('Loss/rr', rr_loss, epoch)
                self.writer.add_scalar('Acc/train', train_acc, epoch)
                self.writer.add_scalar('Time/train', elapsed_time_cur, epoch)

                val_acc, val_loss = self.score(test_dataloader, self.base_criterion)

                # printing acc and loss
                if verbose and (epoch % verbose_after_n_epochs == 0):
                    print(f"Epoch {epoch}| ", end='')
                    print(f"accuracy: {(train_acc):>0.1f}%, loss: {train_loss:>8f} | ", end='')
                    print(f"Test Error: Acc: {val_acc:>0.1f}%, Avg loss: {val_loss:>8f}")
                    #print(f" --number RAWR [{how_many_rawr_epoch}]")

                # test acc on test set
                self.writer.add_scalar('Loss/test', val_loss, epoch)
                self.writer.add_scalar('Acc/test', val_acc, epoch)
                # write in logfile -> we need the logfile to see plots in Jupyter notebooks
                f.write(f"{epoch},{(train_acc):>0.1f},{train_loss:>8f},{ra_loss:>8f},{rr_loss:>8f},{(val_acc):>0.1f},{val_loss:>8f},{elapsed_time_cur:>0.4f}\n")
                
                # log to terminal on switch
                if epoch == disable_xil_loss_first_n_epochs and verbose and disable_xil_loss_first_n_epochs != 0:
                    bs_store = (epoch, train_acc, train_loss, val_acc, val_loss)
 

                # save the current best model on val set
                if save_best and train_loss < best_train_loss:
                    best_train_loss = train_loss
                    self.save_learner(verbose=False, best=True)

                if save_last:
                    self.save_learner(verbose=False)

        print(f"--> Training took {elapsed_time:>4f} seconds!")
        self.writer.flush()
        self.writer.close()

        # return the train_acc and loss after last epoch
        return train_acc, train_loss, elapsed_time, bs_store

    def fit_isic(self, dataloader, test_dataloader, epochs, save_best=False, save_last=True, \
        verbose=True, verbose_after_n_epochs=1, scheduler_=True, alternative_dataloader=None):
        """
        Training loop for the ISIC19 (same than fit() but with alternative_dataloader).
        Only use for no XIL training.
        Fits the learner using training data from dataloader for specified number 
        of epochs. After every epoch the learner is evaluated on the specified
        test_dataloader and the alternative_dataloader. Uses pytorch SummaryWriter 
        to log stats for tensorboard. Writes the training progress and stats per epoch 
        in a logfile to the logs-folder.

        Args:
            dataloader: train dataloader (X, y, expl) where expl are the ground-truth user 
                feedback masks (optional).
            test_dataloader: validation dataloader (Xt, yt).
            epochs: number of epochs to train.
            save_best: saves the best model on the train loss to file.
            save_last: saves the model after every epoch .
            verbose: set to False if you want no print outputs.
            verbose_after_n_epochs: print outputs after every n epochs.
            alternative_dataloader: another dataloader used to evaluate the model after 
                every epoch 
        """
        rtpt = RTPT(name_initials='FF', experiment_name='ISIC-2019-train', max_iterations=epochs)
        rtpt.start()
        print("Start training...")
        with open(f"{self.log_folder}.csv", "w") as f:
            f.write("epoch,acc,loss,ra_loss,rr_loss,val_acc,val_loss,val2_acc,val2_loss,time\n")
            best_train_loss = 100000
            elapsed_time = 0
            if scheduler_:
                scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=8)
                
            for epoch in range(1, epochs+1):
                #how_many_rawr_epoch = torch.tensor([0])

                self.model.train()
                size = len(dataloader.dataset)
                # ra_loss = right answer, rr_loss = right reason 
                train_loss, correct, ra_loss, rr_loss = 0, 0, 0, 0
                start_time = time.time()
                with tqdm(dataloader, unit="batch") as tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                
                    for data in tepoch:
                        self.optimizer.zero_grad()

                        if isinstance(self.loss, RRRLoss):
                            X, y, expl = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                            X.requires_grad_()
                            output = self.model(X)
                            loss, ra_loss_c, rr_loss_c = self.loss(X, y, expl, output)
                            ##### rawr
                            #how_many_rawr_epoch += how_many_rawr.item()
                        
                        elif isinstance(self.loss, RRRGradCamLoss):
                            X, y, expl = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                            X.requires_grad_()
                            output = self.model(X)
                            expl = expl.float()
                            loss, ra_loss_c, rr_loss_c = self.loss(self.model, X, y, expl, output, self.device)
                        
                        elif isinstance(self.loss, CDEPLoss):
                            X, y, expl = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                            output = self.model(X)
                            expl = expl.float()
                            loss, ra_loss_c, rr_loss_c = self.loss(self.model, X, y, expl, output, self.device)

                        elif isinstance(self.loss, HINTLoss):
                            X, y, expl = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                            X.requires_grad_()
                            output = self.model(X)
                            expl = expl.float()
                            loss, ra_loss_c, rr_loss_c = self.loss(self.model, X, y, expl, output, self.device)
                        
                        elif isinstance(self.loss, RBRLoss):
                            X, y, expl = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                            X.requires_grad_()
                            output = self.model(X)
                            expl = expl.float()
                            loss, ra_loss_c, rr_loss_c = self.loss(self.model, X, y, expl, output)

                        else:
                            X, y = data[0].to(self.device), data[1].to(self.device)
                            output = self.model(X)
                            loss = self.loss(output, y)


                        # Backpropagation
                        loss.backward()
                        self.optimizer.step()
                        
                        # for calculating loss, acc per epoch
                        train_loss += loss.item()
                        correct += (output.argmax(1) == y).type(torch.float).sum().item()

                        # for tracking right answer and right reason loss
                        if isinstance(self.loss, (RRRLoss, HINTLoss, CDEPLoss, RBRLoss, RRRGradCamLoss)):
                            ra_loss += ra_loss_c.item()
                            rr_loss += rr_loss_c.item()
                    
                train_loss /= size
                correct /= size
                ra_loss /= size
                rr_loss /= size
                
                train_acc = 100.*correct
                
                end_time = time.time()
                elapsed_time_cur = int(end_time - start_time)
                elapsed_time += int(elapsed_time_cur)

                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/ra', ra_loss, epoch)
                self.writer.add_scalar('Loss/rr', rr_loss, epoch)
                self.writer.add_scalar('Acc/train', train_acc, epoch)
                self.writer.add_scalar('Time/train', elapsed_time_cur, epoch)

                val_acc, val_loss = self.score(test_dataloader, self.base_criterion)
                val2_acc, val2_loss = 0, 0
                if alternative_dataloader is not None:
                    val2_acc, val2_loss = self.score(alternative_dataloader, self.base_criterion)
                if scheduler_:
                    scheduler.step(train_loss)

                # printing acc and loss
                if verbose and (epoch % verbose_after_n_epochs == 0):
                    print(f"Epoch {epoch}| ", end='')
                    print(f"Acc: {(train_acc):>0.1f}%, loss: {train_loss:>8f} | ", end='')
                    print(f"Test: Acc: {val_acc:>0.1f}%, Avg loss: {val_loss:>8f}", end='')
                    if alternative_dataloader is not None:
                        print(f"| Test_NP: Acc: {val2_acc:>0.1f}%", end='')
                    print(f" Time: [{elapsed_time_cur}]s")
                    logging.info(f"Epoch {epoch} | Acc: {(train_acc):>0.1f}%, loss: {train_loss:>8f} | Test: Acc: {val_acc:>0.1f}%, Avg loss: {val_loss:>8f} | Test_NP: Acc: {val2_acc:>0.1f}%, loss: {val2_loss} | Time: [{elapsed_time_cur}]s")

                # test acc on test set
                self.writer.add_scalar('Loss/test', val_loss, epoch)
                self.writer.add_scalar('Acc/test', val_acc, epoch)
                self.writer.add_scalar('Loss/test_2', val2_loss, epoch)
                self.writer.add_scalar('Acc/test_2', val2_acc, epoch)
                # write in logfile -> we need the logfile to see plots in Jupyter notebooks
                f.write(f"{epoch},{(train_acc):>0.1f},{train_loss:>8f},{ra_loss:>8f},{rr_loss:>8f},{(val_acc):>0.1f},{val_loss:>8f},{(val2_acc):>0.1f},{val2_loss:>8f},{elapsed_time_cur:>0.4f}\n")
 

                # save the current best model on val set
                if save_best and train_loss < best_train_loss:
                    best_train_loss = train_loss
                    self.save_learner(epochs=epoch, verbose=False, best=True)

                if save_last:
                    self.save_learner(epochs=epoch, verbose=False)
                rtpt.step()

        print(f"--> Training took {int(elapsed_time)} seconds!")
        self.writer.flush()
        self.writer.close()

        # return the train_acc and loss after last epoch
        return train_acc, train_loss, elapsed_time

    def fit_n_expl_shuffled_dataloader(self, dataloader, test_dataloader, epochs, \
        save_best=False, save_last=True, verbose=True, verbose_after_n_epochs=1, \
            scheduler_=True, alternative_dataloader=None):
        """
        Special fit method that enables to train the model with a limited 
        number of explanations using flag entries in the train dataloader.
        (For MNIST-> Use with n_expl set in the loading of the dataset.)
        (Also used for ISIC19 XIL models)

        Args:
            dataloader: train dataloader (X, y, expl, flags) where expl are the ground-truth user 
                feedback masks, flags indicate instances who have an explanation (1).
            test_dataloader: validation dataloader (Xt, yt).
            epochs: number of epochs to train.
            save_best: saves the best model on the train loss to file.
            save_last: saves the model after every epoch .
            verbose: set to False if you want no print outputs.
            verbose_after_n_epochs: print outputs after every n epochs.
            scheduler_: if True then use scheduler (to adapt scheduler change code below)
            alternative_dataloader: another dataloader used to evaluate the model after 
                every epoch


        Examples:
        ... MNIST
        train_loader, test_loader = decoy_mnist(train_shuffle=True, n_expl=500)
        learner.fit_n_expl_shuffled_dataloader(train_loader, test_loader, epochs=64)

        """
        rtpt = RTPT(name_initials='FF', experiment_name='ISIC-2019-train', max_iterations=epochs)
        rtpt.start()
        print("Start training...")
        with open(f"{self.log_folder}.csv", "w") as f:
            f.write("epoch,n_expl,acc,loss,ra_loss,rr_loss,val_acc,val_loss,val2_acc,val2_loss,time\n")
            best_train_loss = 100000
            elapsed_time = 0

            if scheduler_:
                scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=8)
            for epoch in range(1, epochs+1):

                self.model.train()
                size = len(dataloader.dataset)
                train_loss, correct, ra_loss, rr_loss = 0, 0, 0, 0
                start_time = time.time()
                n_expl = 0

                with tqdm(dataloader, unit="batch") as tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                
                    for data in tepoch:
                        #__import__("pdb").set_trace()

                        self.optimizer.zero_grad()
                        batch_size = data[3].shape[0]
                
                        flags = data[3].to(self.device)
                        # get indices of instances that have flag 1 as these are trained with xil loss 
                        xil_indices = torch.nonzero(flags, as_tuple=True)[0]
                        number_of_used_expl = len(xil_indices)
                        n_expl += number_of_used_expl
                        # find indicices where expl not zero
                        no_xil_indices = torch.nonzero((flags == 0), as_tuple=True)[0]
                        # sanity check
                        if number_of_used_expl + len(no_xil_indices) != len(flags):
                            raise KeyError("xil_indicies and no_xil_indicies do not match flags!")
                        
                        ratio = number_of_used_expl / batch_size

                        X, y = data[0].to(self.device), data[1].to(self.device)

                        if number_of_used_expl == 0:
                            
                            output = self.model(X)
                            loss = self.base_criterion(output, y)
                            # Backpropagation
                            loss.backward()
                            self.optimizer.step()
                            # for calculating loss, acc per epoch
                            train_loss += loss.item()
                            rr_loss_c = torch.zeros(1,)
                            correct += (output.argmax(1) == y).type(torch.float).sum().item()
                            # for tracking right answer and right reason loss
                            if isinstance(self.loss, (RRRLoss, HINTLoss, CDEPLoss, RBRLoss, RRRGradCamLoss)):
                                ra_loss += loss.item()
                                rr_loss += rr_loss_c.item() 
                            
                        else:
                            X_xil, y_xil, expl = data[0][xil_indices].to(self.device), data[1][xil_indices].to(self.device), \
                                data[2][xil_indices].to(self.device)

                            if isinstance(self.loss, RRRLoss):
                                X_xil.requires_grad_()
                                output = self.model(X_xil)
                                loss_xil, ra_loss_c, rr_loss_c = self.loss(X_xil, y_xil, expl, output)
                                
                            elif isinstance(self.loss, RRRGradCamLoss):
                                X_xil.requires_grad_()
                                output = self.model(X_xil)
                                expl = expl.float()
                                loss_xil, ra_loss_c, rr_loss_c = self.loss(self.model, X_xil, y_xil, expl, output, self.device)
                            
                            elif isinstance(self.loss, CDEPLoss):
                                output = self.model(X_xil)
                                expl = expl.float()
                                loss_xil, ra_loss_c, rr_loss_c = self.loss(self.model, X_xil, y_xil, expl, output, self.device)

                            elif isinstance(self.loss, HINTLoss):
                                X_xil.requires_grad_()
                                output = self.model(X_xil)
                                expl = expl.float()
                                loss_xil, ra_loss_c, rr_loss_c = self.loss(self.model, X_xil, y_xil, expl, output, self.device)
                            
                            elif isinstance(self.loss, RBRLoss):
                                X_xil.requires_grad_()
                                output = self.model(X_xil)
                                expl = expl.float()
                                loss_xil, ra_loss_c, rr_loss_c = self.loss(self.model, X_xil, y_xil, expl, output)
                            else:
                                raise Exception

                            # if the whole batch is trained with XIL loss
                            if ratio == 1.:
                                loss = loss_xil
                                if isinstance(self.loss, (RRRLoss, HINTLoss, CDEPLoss, RBRLoss, RRRGradCamLoss)):
                                    ra_loss += ra_loss_c.item()
                                    rr_loss += rr_loss_c.item() 
                            
                            else: # merge right answer loss with rr loss
                                # we only want the rr loss beacuse we calculating the ra loss 
                                # for every example in the batch
                                output = self.model(X)
                                loss_no_xil = self.base_criterion(output, y)

                                loss = rr_loss_c + loss_no_xil
                                # for tracking right answer and right reason loss
                                if isinstance(self.loss, (RRRLoss, HINTLoss, CDEPLoss, RBRLoss, RRRGradCamLoss)):
                                    ra_loss += loss_no_xil.item()
                                    rr_loss += rr_loss_c.item() 

                            # Backpropagation
                            loss.backward()
                            self.optimizer.step()
                        
                            # for calculating loss, acc per epoch
                            train_loss += loss.item()
                            correct += (output.argmax(1) == y).type(torch.float).sum().item()
                    
                train_loss /= size
                correct /= size
                ra_loss /= size
                rr_loss /= size
                
                train_acc = 100.*correct
                
                end_time = time.time()
                elapsed_time_cur = int(end_time - start_time)
                elapsed_time += int(elapsed_time_cur)

                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/ra', ra_loss, epoch)
                self.writer.add_scalar('Loss/rr', rr_loss, epoch)
                self.writer.add_scalar('Acc/train', train_acc, epoch)
                self.writer.add_scalar('Time/train', elapsed_time_cur, epoch)

                val_acc, val_loss = self.score(test_dataloader, self.base_criterion)

                val2_acc, val2_loss = 0, 0
                if alternative_dataloader is not None:
                    val2_acc, val2_loss = self.score(alternative_dataloader, self.base_criterion)

                if scheduler_:
                    scheduler.step(train_loss)

                # printing acc and loss
                if verbose and (epoch % verbose_after_n_epochs == 0):
                    print(f"Epoch {epoch}| ", end='')
                    print(f"Acc: {(train_acc):>0.1f}%, loss: {train_loss:>8f}, rr_loss: {rr_loss} | ", end='')
                    print(f"Test: Acc: {val_acc:>0.1f}%, Avg loss: {val_loss:>8f} | n_expl={n_expl}", end='')
                    if alternative_dataloader is not None:
                        print(f"| Test_NP: Acc: {val2_acc:>0.1f}%", end='')
                    print(f" Time: [{elapsed_time_cur}]s")
                    logging.info(f"Epoch {epoch} | Acc: {(train_acc):>0.1f}%, loss: {train_loss:>8f} | Test: Acc: {val_acc:>0.1f}%, Avg loss: {val_loss:>8f} | Test_NP: Acc: {val2_acc:>0.1f}%, loss: {val2_loss} | Time: [{elapsed_time_cur}]s | n_expl={n_expl}")
                    #print(f" --number RAWR [{how_many_rawr_epoch}]")

                # test acc on test set
                self.writer.add_scalar('Loss/test', val_loss, epoch)
                self.writer.add_scalar('Acc/test', val_acc, epoch)
                self.writer.add_scalar('Loss/test_2', val2_loss, epoch)
                self.writer.add_scalar('Acc/test_2', val2_acc, epoch)
                # write in logfile -> we need the logfile to see plots in Jupyter notebooks
                f.write(f"{epoch},{n_expl},{(train_acc):>0.2f},{train_loss:>12f},{ra_loss:>12f},{rr_loss:>12f},{(val_acc):>0.2f},{val_loss:>12f},{(val2_acc):>0.2f},{val2_loss:>12f},{elapsed_time_cur:>0.4f}\n")

                # save the current best model on val set
                if save_best and train_loss < best_train_loss:
                    best_train_loss = train_loss
                    self.save_learner(epochs=epoch, verbose=False, best=True)

                if save_last:
                    self.save_learner(epochs=epoch, verbose=False)
                
                rtpt.step()

        print(f"--> Training took {int(elapsed_time)} seconds!")
        self.writer.flush()
        self.writer.close()

        # return the train_acc and loss after last epoch
        return train_acc, train_loss, elapsed_time


    def score(self, dataloader, criterion, verbose=False):
        """Returns the acc and loss on the specified dataloader."""
        size = len(dataloader.dataset)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data in dataloader:
                X, y = data[0].to(self.device), data[1].to(self.device)
                output = F.softmax(self.model(X), dim=1)
                test_loss += criterion(output, y).item()
                correct += (output.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size
        if verbose:
            print(f"Test Error: Acc: {100*correct:>0.1f}%, Avg loss: {test_loss:>8f}")
        return 100*correct, test_loss

    def save_learner(self, epochs=0, verbose=1, best=False):
        """Save the model dict to disk."""
        #if best:
        #    self.modelname = self.modelname + "-bestOnTrain"
        results = {'weights': self.model.state_dict(), \
            'optimizer_dict': self.optimizer.state_dict() ,'modelname' : self.modelname, \
            'loss' : str(self.loss), 'epochs': epochs, \
            'rng_state' :  torch.get_rng_state()}
        torch.save(results, 'learner/model_store/' + self.modelname + '.pt')
        if verbose == 1:
            print("Model saved!")

    def load(self, name):
        """Load the model with name from the model_store."""
        checkpoint = torch.load('learner/model_store/'+ name, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['weights'])
        epochs_ = "none"
        if 'optimizer_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_dict'])
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].type(torch.ByteTensor))
        if 'epochs' in checkpoint:
            epochs_ = checkpoint['epochs']
        print(f"Model {self.modelname} loaded! Was trained on {checkpoint['loss']} for {epochs_} epochs!")


    def validation_statistics(self, dataloader, criterion=F.cross_entropy, class_labels=None, \
        save=True, savename="-stats"):
        """Calculates confusion matrix and different statistics on dataloader."""

        size = len(dataloader.dataset)
        y_pred_list = []
        y_val_list = []
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data in dataloader:
                X, y = data[0].to(self.device), data[1].to(self.device)
                output = self.model(X)
                _, preds = output.max(1)
                test_loss += criterion(output, y).item()
                correct += (output.argmax(1) == y).type(torch.float).sum().item()
                y_val_list += y.detach().cpu().tolist()
                y_pred_list += preds.detach().cpu().tolist()
        test_loss /= size
        correct /= size
        
        cm = confusion_matrix(y_val_list, y_pred_list, labels=class_labels)
        cr = classification_report(y_val_list, y_pred_list, labels=class_labels)
        print(f"Acc: {100*correct:>0.1f}%, Avg loss: {test_loss:>8f}")
        print("\nCONFUSION MATRIX ------------\n")
        print(f"{cm}")
        print("\nCLASSIFICATION REPORT ------------\n")
        print(f"{cr}")

        if save:
            with open(f"{self.log_folder}{savename}.txt", "w") as f:
                f.write(f"Acc: {100*correct:>0.1f}%, Avg loss: {test_loss:>8f}\n")
                f.write("Confusion matrix:\n")
                f.write(f"{cm}\n")
                f.write("Classification report:\n")
                f.write(f"{cr}")


    def config_to_string(self):
        return self.modelname + ' -- loss=' + str(self.loss) + ' -- optim=' + str(self.optimizer)  

