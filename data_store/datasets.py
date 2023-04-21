"""Collection of datasets."""
import time
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from torch.utils.data.dataset import ConcatDataset
import pandas as pd
import h5py
from collections import Counter

from data_store.rawdata import load_decoy_mnist
import xil_methods.ce as ce

def decoy_mnist(no_decoy=False, fmnist=False, batch_size=256, device='cuda', \
                train_shuffle=False, test_shuffle=False, hint_expl=False, feedback=None, \
                n_expl=None, flatten=False):
    """
    Load decoy mnist from Ross et. al 2017 and return train and test dataloaders.

    On every image a gray square is added randomly in one of the four corners.
    The gray tone is function of the label in the train set.
    The gray tone of the squares in the test set are random compared to the train set.
    train set:
        number 0 -> square color 0, ... , number 9 -> square color 255
    test set:
        number 0 -> random color, ...,   number 9 -> random
    annonations:
        The colored square for every image.

    Args:
        no_decoy: if True then the original MNIST (FMNIST) without the confounding factors
            is returned.
        fmnist: if True then the decoyFashionMNIST dataset is returned.
        batch_size: specifies the batch size.
        device: either 'cuda' or 'cpu'.
        train_shuffle: Warning...the dataset has a default fixed shuffle, train_shuffle sets the
                        pytorch Dataloader attribute 'shuffle' which 'have the data reshuffled
                        at every epoch'.
        test_shuffle: see train_shuffle.
        hint_expl: annotation masks indicating the correct image region - segments the image.
            Is used for the HINT method.
        feedback: Manipulates the annotation masks. The following options are available...
            'random' -> random masks are returned.
            'adversarial' -> all ones mask.
            'incomplete' -> 50% of the original mask (bottom half of ones).
            'wrong' -> new 5x3 (3x5) rectangle on the top (bottom, left, right) border in
                the middle.
        n_expl: adds n_expl number of true flags to the annotation masks. This makes it
            possible to train a model allowing only n_expl of XIL loss in the Learner class.
            Max value -> 60000.
        flatten: if True then the returned dataloaders contain flattend images (n, 28x28).
        ce : if True then CounterExamples must be added to dataset
    """
    # Xr,Xtr --> orginial MNIST; X,Xt --> decoy MNIST; E,Et --> explanations
    # y,yt --> same for decoy and orginial MNIST
    # note: sets are default flat, t=test
    if fmnist:
        Xr, X, y, E, E_hint, Xtr, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/fashionMnist/decoy-fmnist.npz', fmnist=True)
    else:
        Xr, X, y, E, E_hint, Xtr, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/MNIST/decoy-mnist.npz')

    if feedback is not None:
        n, d = E.shape[0], E.shape[1]
        if feedback == 'random':
            # set explanation feedback masks to random masks
            E = np.random.randint(2, size=n * d).reshape(n, d)
            E_hint = np.random.randint(2, size=n * d).reshape(n, d)

        elif feedback == 'adversarial':
            # set explanation feedback masks to all ones
            E = np.ones((n, d), dtype=np.int64)
            E_hint = np.ones((n, d), dtype=np.int64)

        elif feedback == 'incomplete':
            # delete top half of the feedback mask (4x4 squares -> 2x4 squares)
            for i, e in enumerate(E):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E[i] = e

            for i, e in enumerate(E_hint):
                true_indexes = np.where(e == True)[0]
                half_mask_number = int(true_indexes.size / 2)
                first_half_true_indexes = true_indexes[:half_mask_number]
                np.put(e, first_half_true_indexes, np.array(False))
                E_hint[i] = e

        elif feedback == 'wrong':
            # add a 5 x 3 rectangle in the top middle or bottom middle on the border
            # rectangle is placed on the opposite side of the the confounder square
            if not fmnist:
                for i, e in enumerate(E):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

                if hint_expl:
                    E_hint = np.copy(E)

            if fmnist:
                for i, e in enumerate(E):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

                if hint_expl:
                    E_hint = np.copy(E)

    if n_expl is not None:
        not_used_flags = np.zeros((E.shape[0] - n_expl), dtype=np.int64)
        used_flags = np.ones(n_expl, dtype=np.int64)
        flags = np.concatenate((used_flags, not_used_flags), axis=0)

    if not flatten:  # if input for a conv net sets should not be flat
        n_samples = 60000
        X = X.reshape((n_samples, 1, 28, 28))
        Xt = Xt.reshape((10000, 1, 28, 28))
        E = E.reshape(n_samples, 1, 28, 28)
        E_hint = E_hint.reshape(n_samples, 1, 28, 28)
        Xr = Xr.reshape((n_samples, 1, 28, 28))
        Xtr = Xtr.reshape((10000, 1, 28, 28))
        Et = Et.reshape(10000, 1, 28, 28)

    if device == 'cpu':
        X, y, E, E_hint = torch.from_numpy(X).type(torch.FloatTensor), \
                          torch.from_numpy(y).type(torch.LongTensor), \
                          torch.from_numpy(E).type(torch.LongTensor), \
                          torch.from_numpy(E_hint).type(torch.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.FloatTensor), \
                     torch.from_numpy(yt).type(torch.LongTensor), \
                     torch.from_numpy(Et).type(torch.LongTensor)

        Xr, Xtr = torch.from_numpy(Xr).type(torch.FloatTensor), \
                  torch.from_numpy(Xtr).type(torch.FloatTensor)

    else:
        X, y, E, E_hint = torch.from_numpy(X).type(torch.cuda.FloatTensor), \
                          torch.from_numpy(y).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E_hint).type(torch.cuda.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.cuda.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.cuda.FloatTensor), \
                     torch.from_numpy(yt).type(torch.cuda.LongTensor), \
                     torch.from_numpy(Et).type(torch.cuda.LongTensor)

        Xr, Xtr = torch.from_numpy(Xr).type(torch.cuda.FloatTensor), \
                  torch.from_numpy(Xtr).type(torch.cuda.FloatTensor)

    if no_decoy:
        train, test = TensorDataset(Xr, y), TensorDataset(Xtr, yt, Et)
        # print(f"Train set: {train.shape}")
        # print(f"Test set: {test.shape}")
        return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

    if hint_expl:
        if n_expl is not None:
            train, test = TensorDataset(X, y, E_hint, flags), TensorDataset(Xt, yt, Et)
        else:
            train, test = TensorDataset(X, y, E_hint), TensorDataset(Xt, yt, Et)
        return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

    if n_expl is not None:
        train, test = TensorDataset(X, y, E, flags), TensorDataset(Xt, yt, Et)
    else:
        train, test = TensorDataset(X, y, E), TensorDataset(Xt, yt, Et)

    # print(f"Train set: {train.shape}")
    # print(f"Test set: {test.shape}")

    return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

def decoy_mnist_retrain(elem_num, fmnist=False, batch_size=256, device='cuda', \
                train_shuffle=False, test_shuffle=False, hint_expl=False, feedback=None, \
                n_expl=None, flatten=False):
    if fmnist:
        _, X, y, E, E_hint, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/fashionMnist/decoy-fmnist.npz', fmnist=True)
    else:
        _, X, y, E, E_hint, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/MNIST/decoy-mnist.npz')

    X_retrain = [X[e] for e in elem_num]
    y_retrain = [y[e] for e in elem_num]
    E_retrain = [E[e] for e in elem_num]
    E_hint_retrain = [E_hint[e] for e in elem_num]

    X_retrain = np.array(X_retrain)
    y_retrain = np.array(y_retrain)
    E_retrain = np.array(E_retrain)
    E_hint_retrain = np.array(E_hint_retrain)

    if feedback is not None:
        n, d = E_retrain.shape[0], E_retrain.shape[1]
        if feedback == 'random':
            # set explanation feedback masks to random masks
            E_retrain = np.random.randint(2, size=n * d).reshape(n, d)
            E_hint_retrain = np.random.randint(2, size=n * d).reshape(n, d)

        elif feedback == 'adversarial':
            # set explanation feedback masks to all ones
            E_retrain = np.ones((n, d), dtype=np.int64)
            E_hint_retrain = np.ones((n, d), dtype=np.int64)

        elif feedback == 'incomplete':
            # delete top half of the feedback mask (4x4 squares -> 2x4 squares)
            for i, e in enumerate(E_retrain):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E_retrain[i] = e

            for i, e in enumerate(E_hint_retrain):
                true_indexes = np.where(e == True)[0]
                half_mask_number = int(true_indexes.size / 2)
                first_half_true_indexes = true_indexes[:half_mask_number]
                np.put(e, first_half_true_indexes, np.array(False))
                E_hint_retrain[i] = e

        elif feedback == 'wrong':
            # add a 5 x 3 rectangle in the top middle or bottom middle on the border
            # rectangle is placed on the opposite side of the the confounder square
            if not fmnist:
                for i, e in enumerate(E_retrain):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_retrain[i] = cur

                if hint_expl:
                    E_hint_retrain = np.copy(E_retrain)

            if fmnist:
                for i, e in enumerate(E_retrain):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_retrain[i] = cur

                if hint_expl:
                    E_hint_retrain = np.copy(E_retrain)

    if n_expl is not None:
        not_used_flags = np.zeros((E_retrain.shape[0] - n_expl), dtype=np.int64)
        used_flags = np.ones(n_expl, dtype=np.int64)
        flags = np.concatenate((used_flags, not_used_flags), axis=0)

    if not flatten:  # if input for a conv net sets should not be flat
        n_samples = X_retrain.shape[0]
        X_retrain = X_retrain.reshape((n_samples, 1, 28, 28))
        Xt = Xt.reshape((10000, 1, 28, 28))
        E_retrain = E_retrain.reshape(n_samples, 1, 28, 28)
        E_hint_retrain = E_hint_retrain.reshape(n_samples, 1, 28, 28)
        Et = Et.reshape(10000, 1, 28, 28)

    if device == 'cpu':
        X_retrain, y_retrain, E_retrain, E_hint_retrain = torch.from_numpy(X_retrain).type(torch.FloatTensor), \
                          torch.from_numpy(y_retrain).type(torch.LongTensor), \
                          torch.from_numpy(E_retrain).type(torch.LongTensor), \
                          torch.from_numpy(E_hint_retrain).type(torch.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.FloatTensor), \
                     torch.from_numpy(yt).type(torch.LongTensor), \
                     torch.from_numpy(Et).type(torch.LongTensor)

    else:
        X_retrain, y_retrain, E_retrain, E_hint_retrain = torch.from_numpy(X_retrain).type(torch.cuda.FloatTensor), \
                          torch.from_numpy(y_retrain).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E_retrain).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E_hint_retrain).type(torch.cuda.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.cuda.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.cuda.FloatTensor), \
                     torch.from_numpy(yt).type(torch.cuda.LongTensor), \
                     torch.from_numpy(Et).type(torch.cuda.LongTensor)

    if hint_expl:
        if n_expl is not None:
            train, test = TensorDataset(X_retrain, y_retrain, E_hint_retrain, flags), TensorDataset(Xt, yt, Et)
        else:
            train, test = TensorDataset(X_retrain, y_retrain, E_hint_retrain), TensorDataset(Xt, yt, Et)
        return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

    if n_expl is not None:
        train, test = TensorDataset(X_retrain, y_retrain, E_retrain, flags), TensorDataset(Xt, yt, Et)
    else:
        train, test = TensorDataset(X_retrain, y_retrain, E_retrain), TensorDataset(Xt, yt, Et)

    # print(f"Train set: {train.shape}")
    # print(f"Test set: {test.shape}")

    return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

def decoy_mnist_both(no_decoy=False, fmnist=False, batch_size=256, device='cuda', \
    train_shuffle=False, test_shuffle=False, feedback=None, \
    n_expl=None, flatten=False):
    if fmnist:
        Xr, X, y, E, E_hint, Xtr, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/fashionMnist/decoy-fmnist.npz', fmnist=True)
    else:
        Xr, X, y, E, E_hint, Xtr, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/MNIST/decoy-mnist.npz')

    if feedback is not None:
        n, d = E.shape[0], E.shape[1]
        if feedback == 'random':
            # set explanation feedback masks to random masks
            E = np.random.randint(2, size=n * d).reshape(n, d)
            E_hint = np.random.randint(2, size=n * d).reshape(n, d)

        elif feedback == 'adversarial':
            # set explanation feedback masks to all ones
            E = np.ones((n, d), dtype=np.int64)
            E_hint = np.ones((n, d), dtype=np.int64)

        elif feedback == 'incomplete':
            # delete top half of the feedback mask (4x4 squares -> 2x4 squares)
            for i, e in enumerate(E):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E[i] = e

            for i, e in enumerate(E_hint):
                true_indexes = np.where(e == True)[0]
                half_mask_number = int(true_indexes.size / 2)
                first_half_true_indexes = true_indexes[:half_mask_number]
                np.put(e, first_half_true_indexes, np.array(False))
                E_hint[i] = e

        elif feedback == 'wrong':
            # add a 5 x 3 rectangle in the top middle or bottom middle on the border
            # rectangle is placed on the opposite side of the the confounder square
            if not fmnist:
                for i, e in enumerate(E):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

                E_hint = np.copy(E)

            if fmnist:
                for i, e in enumerate(E):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

                E_hint = np.copy(E)

    if n_expl is not None:
        not_used_flags = np.zeros((E.shape[0] - n_expl), dtype=np.int64)
        used_flags = np.ones(n_expl, dtype=np.int64)
        flags = np.concatenate((used_flags, not_used_flags), axis=0)

    if not flatten:  # if input for a conv net sets should not be flat
        X = X.reshape((60000, 1, 28, 28))
        Xt = Xt.reshape((10000, 1, 28, 28))
        E = E.reshape(60000, 1, 28, 28)
        E_hint = E_hint.reshape(60000, 1, 28, 28)
        Xr = Xr.reshape((60000, 1, 28, 28))
        Xtr = Xtr.reshape((10000, 1, 28, 28))
        Et = Et.reshape(10000, 1, 28, 28)

    if device == 'cpu':
        X, y, E, E_hint = torch.from_numpy(X).type(torch.FloatTensor), \
                          torch.from_numpy(y).type(torch.LongTensor), \
                          torch.from_numpy(E).type(torch.LongTensor), \
                          torch.from_numpy(E_hint).type(torch.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.FloatTensor), \
                     torch.from_numpy(yt).type(torch.LongTensor), \
                     torch.from_numpy(Et).type(torch.LongTensor)

        Xr, Xtr = torch.from_numpy(Xr).type(torch.FloatTensor), \
                  torch.from_numpy(Xtr).type(torch.FloatTensor)

    else:
        X, y, E, E_hint = torch.from_numpy(X).type(torch.cuda.FloatTensor), \
                          torch.from_numpy(y).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E_hint).type(torch.cuda.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.cuda.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.cuda.FloatTensor), \
                     torch.from_numpy(yt).type(torch.cuda.LongTensor), \
                     torch.from_numpy(Et).type(torch.cuda.LongTensor)

        Xr, Xtr = torch.from_numpy(Xr).type(torch.cuda.FloatTensor), \
                  torch.from_numpy(Xtr).type(torch.cuda.FloatTensor)

    if no_decoy:
        train, test = TensorDataset(Xr, y), TensorDataset(Xtr, yt, Et)
        # print(f"Train set: {train.shape}")
        # print(f"Test set: {test.shape}")
        return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

    if n_expl is not None:
        train, test = TensorDataset(X, y, E, E_hint, flags), TensorDataset(Xt, yt, Et)
    else:
        train, test = TensorDataset(X, y, E, E_hint), TensorDataset(Xt, yt, Et)

    return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

def decoy_mnist_both_retrain(elem_num, fmnist=False, batch_size=256, device='cuda', \
                train_shuffle=False, test_shuffle=False, hint_expl=False, feedback=None, \
                n_expl=None, flatten=False):
    if fmnist:
        _, X, y, E, E_hint, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/fashionMnist/decoy-fmnist.npz', fmnist=True)
    else:
        _, X, y, E, E_hint, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/MNIST/decoy-mnist.npz')

    X_retrain = [X[e] for e in elem_num]
    y_retrain = [y[e] for e in elem_num]
    E_retrain = [E[e] for e in elem_num]
    E_hint_retrain = [E_hint[e] for e in elem_num]

    X_retrain = np.array(X_retrain)
    y_retrain = np.array(y_retrain)
    E_retrain = np.array(E_retrain)
    E_hint_retrain = np.array(E_hint_retrain)

    if feedback is not None:
        n, d = E_retrain.shape[0], E_retrain.shape[1]
        if feedback == 'random':
            # set explanation feedback masks to random masks
            E_retrain = np.random.randint(2, size=n * d).reshape(n, d)
            E_hint_retrain = np.random.randint(2, size=n * d).reshape(n, d)

        elif feedback == 'adversarial':
            # set explanation feedback masks to all ones
            E_retrain = np.ones((n, d), dtype=np.int64)
            E_hint_retrain = np.ones((n, d), dtype=np.int64)

        elif feedback == 'incomplete':
            # delete top half of the feedback mask (4x4 squares -> 2x4 squares)
            for i, e in enumerate(E_retrain):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E_retrain[i] = e

            for i, e in enumerate(E_hint_retrain):
                true_indexes = np.where(e == True)[0]
                half_mask_number = int(true_indexes.size / 2)
                first_half_true_indexes = true_indexes[:half_mask_number]
                np.put(e, first_half_true_indexes, np.array(False))
                E_hint_retrain[i] = e

        elif feedback == 'wrong':
            # add a 5 x 3 rectangle in the top middle or bottom middle on the border
            # rectangle is placed on the opposite side of the the confounder square
            if not fmnist:
                for i, e in enumerate(E_retrain):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_retrain[i] = cur

                if hint_expl:
                    E_hint_retrain = np.copy(E_retrain)

            if fmnist:
                for i, e in enumerate(E_retrain):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_retrain[i] = cur

                if hint_expl:
                    E_hint_retrain = np.copy(E_retrain)

    if n_expl is not None:
        not_used_flags = np.zeros((E_retrain.shape[0] - n_expl), dtype=np.int64)
        used_flags = np.ones(n_expl, dtype=np.int64)
        flags = np.concatenate((used_flags, not_used_flags), axis=0)

    if not flatten:  # if input for a conv net sets should not be flat
        n_samples = X_retrain.shape[0]
        X_retrain = X_retrain.reshape((n_samples, 1, 28, 28))
        Xt = Xt.reshape((10000, 1, 28, 28))
        E_retrain = E_retrain.reshape(n_samples, 1, 28, 28)
        E_hint_retrain = E_hint_retrain.reshape(n_samples, 1, 28, 28)
        Et = Et.reshape(10000, 1, 28, 28)

    if device == 'cpu':
        X_retrain, y_retrain, E_retrain, E_hint_retrain = torch.from_numpy(X_retrain).type(torch.FloatTensor), \
                          torch.from_numpy(y_retrain).type(torch.LongTensor), \
                          torch.from_numpy(E_retrain).type(torch.LongTensor), \
                          torch.from_numpy(E_hint_retrain).type(torch.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.FloatTensor), \
                     torch.from_numpy(yt).type(torch.LongTensor), \
                     torch.from_numpy(Et).type(torch.LongTensor)

    else:
        X_retrain, y_retrain, E_retrain, E_hint_retrain = torch.from_numpy(X_retrain).type(torch.cuda.FloatTensor), \
                          torch.from_numpy(y_retrain).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E_retrain).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E_hint_retrain).type(torch.cuda.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.cuda.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.cuda.FloatTensor), \
                     torch.from_numpy(yt).type(torch.cuda.LongTensor), \
                     torch.from_numpy(Et).type(torch.cuda.LongTensor)

    if n_expl is not None:
        train, test = TensorDataset(X_retrain, y_retrain, E_retrain, E_hint_retrain, flags), TensorDataset(Xt, yt, Et)
    else:
        train, test = TensorDataset(X_retrain, y_retrain, E_retrain, E_hint_retrain), TensorDataset(Xt, yt, Et)

    return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

    # print(f"Train set: {train.shape}")
    # print(f"Test set: {test.shape}")

    return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

def decoy_mnist_CE_augmented(fmnist=False, batch_size=256, device='cuda', \
    train_shuffle=False, test_shuffle=False, n_instances=-1, n_counterexamples_per_instance=1, \
        ce_strategy='random', feedback=None, flatten=False,):
    """
    Load CE augemented decoy mnist from Ross et. al 2017 and return train and test dataloader.
    The CE method adds counterexamples to the train set which are based on the 
    provided ground-truth annotation masks.
    See: https://dl.acm.org/doi/10.1145/3306618.3314293.  

    Args:
        fmnist: if True then the decoyFashionMNIST dataset is returned.
        batch_size: specifies the batch size.
        device: either 'cuda' or 'cpu'.
        train_shuffle: Warning...the dataset has a default fixed shuffle, train_shuffle sets the
                        pytorch Dataloader attribute 'shuffle' which 'have the data reshuffled 
                        at every epoch'.
        test_shuffle: see train_shuffle.
        n_instances: Number of counterexamples which are added to the train set (max 60000).
                    If -1 then for every image counterexamples are added 
                        -> doubles the size of train set.
        n_counterexamples_per_instance: Number of counterexamples for one instances (1,2,3).
        ce_strategy: Augmentation strategy, default 'random' (currently only randomization
            of confounder pixels is supported).
        feedback: see decoy_mnist() above.
        flatten: if True then the returned dataloaders contain flattend images (n, 28x28).
    """
    # Xr,Xtr --> orginial MNIST; X,Xt --> decoy MNIST; E,Et --> explanations
    # y,yt --> same for decoy and orginial MNIST
    # note: sets are default flat, t=test 
    if fmnist:
        _, X, y, E, _, _, Xt, yt, _, _ = load_decoy_mnist.generate_dataset(\
            cachefile='data_store/rawdata/fashionMnist/decoy-fmnist.npz', fmnist=True)
    else:
        _, X, y, E, _, _, Xt, yt, _, _ = load_decoy_mnist.generate_dataset(\
            cachefile='data_store/rawdata/MNIST/decoy-mnist.npz')

    if feedback is not None:
        n, d = E.shape[0], E.shape[1]
        if feedback == 'random':
            # set explanation feedback masks to random masks
            E = np.random.randint(2, size=n*d).reshape(n, d).astype(bool)

        elif feedback == 'adversarial':
            # set explanation feedback masks to random masks
            E = np.ones((n,d), dtype=np.int64).astype(bool)

        elif feedback == 'incomplete':
            # delete top half of the feedback mask (4x4 squares -> 2x4 squares)
            for i, e in enumerate(E):
                true_indexes = np.where(e==True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E[i] = e.astype(bool)

        elif feedback == 'wrong':
            # add a 5 x 3 rectangle in the top middle or bottom middle on the border
            # rectangle is placed on the opposite side of the the confounder square
            if not fmnist:
                for i, e in enumerate(E):
                    det = np.where(e==True)[0]
                    if det.item(0) in [x for x in range(0,28)]: # top row
                        ind = [738,739,740,741,742,743,744,745,766,767,768,769,770,771,772,773]
                    else: # bottom row
                        ind = [10,11,12,13,14,15,16,17,38,39,40,41,42,43,44,45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur
            if fmnist:
                for i, e in enumerate(E):
                    det = np.where(e==True)[0]
                    if y[i] in [0,1,2,3,4,6,8]: # rectangle left right
                        if det.item(0) in [0, 672]: # left
                            ind = [306,307,334,335,362,363,390,391,418,419,446,447,474,475,502,503]
                        else: # right
                            ind = [280,281,308,309,336,337,364,365,392,393,420,421,448,449,476,477]
                    else: # rectangle bottom top row
                        if det.item(0) in [0, 24]: # top
                            ind = [738,739,740,741,742,743,744,745,766,767,768,769,770,771,772,773]
                        else: # bottom
                            ind = [10,11,12,13,14,15,16,17,38,39,40,41,42,43,44,45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

    # augment trainset according to specified params
    X_corrections, y_corrections = ce.get_corrections(X, E, y, n_instances, \
        n_counterexamples_per_instance, ce_strategy)
    # add to train set    
    X = np.vstack([X, X_corrections])
    y = np.hstack([y, y_corrections])
    print(f"Train set was augmented: X.size= {len(X)}, y.size= {len(y)}")


    if not flatten: # if input for a conv net sets should not be flat
        n_samples = X.shape[0]
        X = X.reshape((n_samples, 1, 28, 28))
        Xt = Xt.reshape((10000, 1, 28, 28))
        # for current experiments we dont need the explanations anymore, 
        # but if we need the expl we have to expand the expl set with all zeros masks for the last 
        # [len(X_corrections) - 60000] 
        #E = E[:60000]
        #E = E.reshape(60000, 1, 28, 28)

    if device == 'cpu':
        X, y = torch.from_numpy(X).type(torch.FloatTensor), \
            torch.from_numpy(y).type(torch.LongTensor), \

        Xt, yt = torch.from_numpy(Xt).type(torch.FloatTensor), \
            torch.from_numpy(yt).type(torch.LongTensor)

    else:
        X, y = torch.from_numpy(X).type(torch.cuda.FloatTensor), \
            torch.from_numpy(y).type(torch.cuda.LongTensor), \

        Xt, yt = torch.from_numpy(Xt).type(torch.cuda.FloatTensor), \
            torch.from_numpy(yt).type(torch.cuda.LongTensor)

    breakpoint()
    train, test = TensorDataset(X, y), TensorDataset(Xt, yt)
    return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

def decoy_mnist_CE_augmented_retrain(elem_num, fmnist=False, batch_size=256, device='cuda', \
    train_shuffle=False, test_shuffle=False, n_instances=-1, n_counterexamples_per_instance=1, \
        ce_strategy='random', feedback=None, flatten=False,):
    """
    Load CE augemented decoy mnist from Ross et. al 2017 and return train and test dataloader.
    The CE method adds counterexamples to the train set which are based on the
    provided ground-truth annotation masks.
    See: https://dl.acm.org/doi/10.1145/3306618.3314293.

    Args:
        fmnist: if True then the decoyFashionMNIST dataset is returned.
        batch_size: specifies the batch size.
        device: either 'cuda' or 'cpu'.
        train_shuffle: Warning...the dataset has a default fixed shuffle, train_shuffle sets the
                        pytorch Dataloader attribute 'shuffle' which 'have the data reshuffled
                        at every epoch'.
        test_shuffle: see train_shuffle.
        n_instances: Number of counterexamples which are added to the train set (max 60000).
                    If -1 then for every image counterexamples are added
                        -> doubles the size of train set.
        n_counterexamples_per_instance: Number of counterexamples for one instances (1,2,3).
        ce_strategy: Augmentation strategy, default 'random' (currently only randomization
            of confounder pixels is supported).
        feedback: see decoy_mnist() above.
        flatten: if True then the returned dataloaders contain flattend images (n, 28x28).
    """
    # Xr,Xtr --> orginial MNIST; X,Xt --> decoy MNIST; E,Et --> explanations
    # y,yt --> same for decoy and orginial MNIST
    # note: sets are default flat, t=test
    if fmnist:
        _, X, y, E, _, _, Xt, yt, _, _ = load_decoy_mnist.generate_dataset(\
            cachefile='data_store/rawdata/fashionMnist/decoy-fmnist.npz', fmnist=True)
    else:
        _, X, y, E, _, _, Xt, yt, _, _ = load_decoy_mnist.generate_dataset(\
            cachefile='data_store/rawdata/MNIST/decoy-mnist.npz')

    if feedback is not None:
        n, d = E.shape[0], E.shape[1]
        if feedback == 'random':
            # set explanation feedback masks to random masks
            E = np.random.randint(2, size=n*d).reshape(n, d).astype(bool)

        elif feedback == 'adversarial':
            # set explanation feedback masks to random masks
            E = np.ones((n,d), dtype=np.int64).astype(bool)

        elif feedback == 'incomplete':
            # delete top half of the feedback mask (4x4 squares -> 2x4 squares)
            for i, e in enumerate(E):
                true_indexes = np.where(e==True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E[i] = e.astype(bool)

        elif feedback == 'wrong':
            # add a 5 x 3 rectangle in the top middle or bottom middle on the border
            # rectangle is placed on the opposite side of the the confounder square
            if not fmnist:
                for i, e in enumerate(E):
                    det = np.where(e==True)[0]
                    if det.item(0) in [x for x in range(0,28)]: # top row
                        ind = [738,739,740,741,742,743,744,745,766,767,768,769,770,771,772,773]
                    else: # bottom row
                        ind = [10,11,12,13,14,15,16,17,38,39,40,41,42,43,44,45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur
            if fmnist:
                for i, e in enumerate(E):
                    det = np.where(e==True)[0]
                    if y[i] in [0,1,2,3,4,6,8]: # rectangle left right
                        if det.item(0) in [0, 672]: # left
                            ind = [306,307,334,335,362,363,390,391,418,419,446,447,474,475,502,503]
                        else: # right
                            ind = [280,281,308,309,336,337,364,365,392,393,420,421,448,449,476,477]
                    else: # rectangle bottom top row
                        if det.item(0) in [0, 24]: # top
                            ind = [738,739,740,741,742,743,744,745,766,767,768,769,770,771,772,773]
                        else: # bottom
                            ind = [10,11,12,13,14,15,16,17,38,39,40,41,42,43,44,45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

    # augment trainset according to specified params
    X_corrections, y_corrections = ce.get_corrections(X, E, y, n_instances, \
        n_counterexamples_per_instance, ce_strategy)
    # add to train set
    X = np.vstack([X, X_corrections])
    y = np.hstack([y, y_corrections])
    print(f"Train set was augmented: X.size= {len(X)}, y.size= {len(y)}")

    X_retrain = [X[e] for e in elem_num]
    y_retrain = [y[e] for e in elem_num]

    X_retrain = np.array(X_retrain)
    y_retrain = np.array(y_retrain)

    if not flatten: # if input for a conv net sets should not be flat
        n_samples = X_retrain.shape[0]
        X_retrain = X_retrain.reshape((n_samples, 1, 28, 28))
        Xt = Xt.reshape((10000, 1, 28, 28))
        # for current experiments we dont need the explanations anymore,
        # but if we need the expl we have to expand the expl set with all zeros masks for the last
        # [len(X_corrections) - 60000]
        #E = E[:60000]
        #E = E.reshape(60000, 1, 28, 28)

    if device == 'cpu':
        X_retrain, y_retrain = torch.from_numpy(X_retrain).type(torch.FloatTensor), \
            torch.from_numpy(y_retrain).type(torch.LongTensor), \

        Xt, yt = torch.from_numpy(Xt).type(torch.FloatTensor), \
            torch.from_numpy(yt).type(torch.LongTensor)

    else:
        X_retrain, y_retrain = torch.from_numpy(X_retrain).type(torch.cuda.FloatTensor), \
            torch.from_numpy(y_retrain).type(torch.cuda.LongTensor), \

        Xt, yt = torch.from_numpy(Xt).type(torch.cuda.FloatTensor), \
            torch.from_numpy(yt).type(torch.cuda.LongTensor)

    train, test = TensorDataset(X_retrain, y_retrain), TensorDataset(Xt, yt)
    return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

def decoy_mnist_CE_combined(fmnist=False, batch_size=256, device='cuda', \
                train_shuffle=False, test_shuffle=False, hint_expl=False, feedback=None, \
                n_expl=None, flatten=False, n_instances=-1, n_counterexamples_per_instance=1, ce_strategy='random'):
    if fmnist:
        _, X, y, E, E_hint, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/fashionMnist/decoy-fmnist.npz', fmnist=True)
    else:
        _, X, y, E, E_hint, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/MNIST/decoy-mnist.npz')

    E_ce = E

    if feedback is not None:
        n, d = E.shape[0], E.shape[1]
        if feedback == 'random':
            # set explanation feedback masks to random masks
            E = np.random.randint(2, size=n * d).reshape(n, d)
            E_ce = np.random.randint(2, size=n*d).reshape(n, d).astype(bool)
            E_hint = np.random.randint(2, size=n * d).reshape(n, d)

        elif feedback == 'adversarial':
            # set explanation feedback masks to all ones
            E = np.ones((n, d), dtype=np.int64)
            E_ce = np.ones((n,d), dtype=np.int64).astype(bool)
            E_hint = np.ones((n, d), dtype=np.int64)

        elif feedback == 'incomplete':
            # delete top half of the feedback mask (4x4 squares -> 2x4 squares)
            for i, e in enumerate(E):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E[i] = e

            for i, e in enumerate(E_ce):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E_ce[i] = e.astype(bool)

            for i, e in enumerate(E_hint):
                true_indexes = np.where(e == True)[0]
                half_mask_number = int(true_indexes.size / 2)
                first_half_true_indexes = true_indexes[:half_mask_number]
                np.put(e, first_half_true_indexes, np.array(False))
                E_hint[i] = e

        elif feedback == 'wrong':
            # add a 5 x 3 rectangle in the top middle or bottom middle on the border
            # rectangle is placed on the opposite side of the the confounder square
            if not fmnist:
                for i, e in enumerate(E):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

                for i, e in enumerate(E_ce):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_ce[i] = cur

                if hint_expl:
                    E_hint = np.copy(E)

            if fmnist:
                for i, e in enumerate(E):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

                for i, e in enumerate(E_ce):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_ce[i] = cur

                if hint_expl:
                    E_hint = np.copy(E)

    X_corrections, y_corrections = ce.get_corrections(X, E_ce, y, n_instances, n_counterexamples_per_instance, ce_strategy)
    # add to train set
    X = np.vstack([X, X_corrections])
    y = np.hstack([y, y_corrections])
    print(f"Train set was augmented: X.size= {len(X)}, y.size= {len(y)}")
    one = np.ones(60000, dtype=int)  # first 60000 are non CE
    zero = np.zeros(60000, dtype=int) # next 60000 are CE
    mask = np.append(one, zero) # 0 = non CE, 1 = CE

    E_hint = np.append(E_hint, E_ce)
    E = np.append(E, E_ce)

    if n_expl is not None:
        not_used_flags = np.zeros((E.shape[0] - n_expl), dtype=np.int64)
        used_flags = np.ones(n_expl, dtype=np.int64)
        flags = np.concatenate((used_flags, not_used_flags), axis=0)

    if not flatten:  # if input for a conv net sets should not be flat
        n_samples = X.shape[0]

        X = X.reshape((n_samples, 1, 28, 28))
        Xt = Xt.reshape((10000, 1, 28, 28))
        E = E.reshape(n_samples, 1, 28, 28)
        E_hint = E_hint.reshape(n_samples, 1, 28, 28)
        Et = Et.reshape(10000, 1, 28, 28)

    if device == 'cpu':
        X, y, E, E_hint = torch.from_numpy(X).type(torch.FloatTensor), \
                          torch.from_numpy(y).type(torch.LongTensor), \
                          torch.from_numpy(E).type(torch.LongTensor), \
                          torch.from_numpy(E_hint).type(torch.FloatTensor)

        mask = torch.from_numpy(mask).type(torch.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.FloatTensor), \
                     torch.from_numpy(yt).type(torch.LongTensor), \
                     torch.from_numpy(Et).type(torch.LongTensor)

    else:
        X, y, E, E_hint = torch.from_numpy(X).type(torch.cuda.FloatTensor), \
                          torch.from_numpy(y).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E_hint).type(torch.cuda.FloatTensor)

        mask = torch.from_numpy(mask).type(torch.cuda.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.cuda.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.cuda.FloatTensor), \
                     torch.from_numpy(yt).type(torch.cuda.LongTensor), \
                     torch.from_numpy(Et).type(torch.cuda.LongTensor)

    if hint_expl:
        if n_expl is not None:
            train, test = TensorDataset(X, y, E_hint, mask, flags), TensorDataset(Xt, yt, Et)
        else:
            train, test = TensorDataset(X, y, E_hint, mask), TensorDataset(Xt, yt, Et)
        return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

    if n_expl is not None:
        train, test = TensorDataset(X, y, E, mask, flags), TensorDataset(Xt, yt, Et)
    else:
        train, test = TensorDataset(X, y, E, mask), TensorDataset(Xt, yt, Et)

    # print(f"Train set: {train.shape}")
    # print(f"Test set: {test.shape}")
    return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

def decoy_mnist_CE_combined_retrain(elem_num, fmnist=False, batch_size=256, device='cuda', \
                train_shuffle=False, test_shuffle=False, hint_expl=False, feedback=None, \
                n_expl=None, flatten=False, n_instances=-1, n_counterexamples_per_instance=1, ce_strategy='random'):
    if fmnist:
        _, X, y, E, E_hint, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/fashionMnist/decoy-fmnist.npz', fmnist=True)
    else:
        _, X, y, E, E_hint, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/MNIST/decoy-mnist.npz')

    E_ce = E

    if feedback is not None:
        n, d = E.shape[0], E.shape[1]
        if feedback == 'random':
            # set explanation feedback masks to random masks
            E = np.random.randint(2, size=n * d).reshape(n, d)
            E_ce = np.random.randint(2, size=n*d).reshape(n, d).astype(bool)
            E_hint = np.random.randint(2, size=n * d).reshape(n, d)

        elif feedback == 'adversarial':
            # set explanation feedback masks to all ones
            E = np.ones((n, d), dtype=np.int64)
            E_ce = np.ones((n,d), dtype=np.int64).astype(bool)
            E_hint = np.ones((n, d), dtype=np.int64)

        elif feedback == 'incomplete':
            # delete top half of the feedback mask (4x4 squares -> 2x4 squares)
            for i, e in enumerate(E):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E[i] = e

            for i, e in enumerate(E_ce):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E_ce[i] = e.astype(bool)

            for i, e in enumerate(E_hint):
                true_indexes = np.where(e == True)[0]
                half_mask_number = int(true_indexes.size / 2)
                first_half_true_indexes = true_indexes[:half_mask_number]
                np.put(e, first_half_true_indexes, np.array(False))
                E_hint[i] = e

        elif feedback == 'wrong':
            # add a 5 x 3 rectangle in the top middle or bottom middle on the border
            # rectangle is placed on the opposite side of the the confounder square
            if not fmnist:
                for i, e in enumerate(E):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

                for i, e in enumerate(E_ce):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_ce[i] = cur

                if hint_expl:
                    E_hint = np.copy(E)

            if fmnist:
                for i, e in enumerate(E):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

                for i, e in enumerate(E_ce):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_ce[i] = cur

                if hint_expl:
                    E_hint = np.copy(E)

    X_corrections, y_corrections = ce.get_corrections(X, E_ce, y, n_instances, n_counterexamples_per_instance, ce_strategy)
    # add to train set
    X = np.vstack([X, X_corrections])
    y = np.hstack([y, y_corrections])
    print(f"Train set was augmented: X.size= {len(X)}, y.size= {len(y)}")
    one = np.ones(60000, dtype=int)  # first 60000 are non CE
    zero = np.zeros(60000, dtype=int) # next 60000 are CE
    mask = np.append(one, zero) # 0 = non CE, 1 = CE

    E_hint = np.append(E_hint, E_ce)
    E = np.append(E, E_ce)

    E_hint_retrain = [E_hint[e] for e in elem_num]
    E_retrain = [E[e] for e in elem_num]
    X_retrain = [X[e] for e in elem_num]
    y_retrain = [y[e] for e in elem_num]
    mask_retrain = [mask[e] for e in elem_num]

    X_retrain = np.array(X_retrain)
    y_retrain = np.array(y_retrain)
    E_retrain = np.array(E_retrain)
    E_hint_retrain = np.array(E_hint_retrain)
    mask_retrain = np.array(mask_retrain)


    if n_expl is not None:
        not_used_flags = np.zeros((E_retrain.shape[0] - n_expl), dtype=np.int64)
        used_flags = np.ones(n_expl, dtype=np.int64)
        flags = np.concatenate((used_flags, not_used_flags), axis=0)

    if not flatten:  # if input for a conv net sets should not be flat
        n_samples = X_retrain.shape[0]

        X = X.reshape((n_samples, 1, 28, 28))
        Xt = Xt.reshape((10000, 1, 28, 28))
        E = E.reshape(n_samples, 1, 28, 28)
        E_hint = E_hint.reshape(n_samples, 1, 28, 28)
        Et = Et.reshape(10000, 1, 28, 28)

    if device == 'cpu':
        X_retrain, y_retrain, E_retrain, E_hint_retrain = torch.from_numpy(X_retrain).type(torch.FloatTensor), \
                          torch.from_numpy(y_retrain).type(torch.LongTensor), \
                          torch.from_numpy(E_retrain).type(torch.LongTensor), \
                          torch.from_numpy(E_hint_retrain).type(torch.FloatTensor)

        mask = torch.from_numpy(mask_retrain).type(torch.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.FloatTensor), \
                     torch.from_numpy(yt).type(torch.LongTensor), \
                     torch.from_numpy(Et).type(torch.LongTensor)


    else:
        X_retrain, y_retrain, E_retrain, E_hint_retrain = torch.from_numpy(X_retrain).type(torch.cuda.FloatTensor), \
                          torch.from_numpy(y_retrain).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E_retrain).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E_hint_retrain).type(torch.cuda.FloatTensor)

        mask_retrain = torch.from_numpy(mask_retrain).type(torch.cuda.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.cuda.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.cuda.FloatTensor), \
                     torch.from_numpy(yt).type(torch.cuda.LongTensor), \
                     torch.from_numpy(Et).type(torch.cuda.LongTensor)

    if hint_expl:
        if n_expl is not None:
            train, test = TensorDataset(X_retrain, y_retrain, E_hint_retrain, mask_retrain, flags), TensorDataset(Xt, yt, Et)
        else:
            train, test = TensorDataset(X_retrain, y_retrain, E_hint_retrain, mask_retrain), TensorDataset(Xt, yt, Et)
        return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

    if n_expl is not None:
        train, test = TensorDataset(X_retrain, y_retrain, E_retrain, mask_retrain, flags), TensorDataset(Xt, yt, Et)
    else:
        train, test = TensorDataset(X_retrain, y_retrain, E_retrain, mask_retrain), TensorDataset(Xt, yt, Et)

    # print(f"Train set: {train.shape}")
    # print(f"Test set: {test.shape}")
    return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

def decoy_mnist_all(fmnist=False, batch_size=256, device='cuda', train_shuffle=False, \
                    test_shuffle=False,feedback=None, n_expl=None, flatten=False, \
                    n_instances=-1, n_counterexamples_per_instance=1, counter_ex=False, ce_strategy='random'):
    if fmnist:
        _, X, y, E, E_hint, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/fashionMnist/decoy-fmnist.npz', fmnist=True)
    else:
        _, X, y, E, E_hint, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/MNIST/decoy-mnist.npz')


    E_ce = E
    mask = np.ones(60000, dtype=int)

    if feedback is not None:
        n, d = E.shape[0], E.shape[1]
        if feedback == 'random':
            # set explanation feedback masks to random masks
            E = np.random.randint(2, size=n * d).reshape(n, d)
            E_ce = np.random.randint(2, size=n*d).reshape(n, d).astype(bool)
            E_hint = np.random.randint(2, size=n * d).reshape(n, d)

        elif feedback == 'adversarial':
            # set explanation feedback masks to all ones
            E = np.ones((n, d), dtype=np.int64)
            E_ce = np.ones((n,d), dtype=np.int64).astype(bool)
            E_hint = np.ones((n, d), dtype=np.int64)

        elif feedback == 'incomplete':
            # delete top half of the feedback mask (4x4 squares -> 2x4 squares)
            for i, e in enumerate(E):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E[i] = e

            for i, e in enumerate(E_ce):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E_ce[i] = e.astype(bool)

            for i, e in enumerate(E_hint):
                true_indexes = np.where(e == True)[0]
                half_mask_number = int(true_indexes.size / 2)
                first_half_true_indexes = true_indexes[:half_mask_number]
                np.put(e, first_half_true_indexes, np.array(False))
                E_hint[i] = e

        elif feedback == 'wrong':
            # add a 5 x 3 rectangle in the top middle or bottom middle on the border
            # rectangle is placed on the opposite side of the the confounder square
            if not fmnist:
                for i, e in enumerate(E):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

                for i, e in enumerate(E_ce):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_ce[i] = cur

                E_hint = np.copy(E)

            if fmnist:
                for i, e in enumerate(E):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E[i] = cur

                for i, e in enumerate(E_ce):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_ce[i] = cur

                E_hint = np.copy(E)

    if counter_ex:
        X_corrections, y_corrections = ce.get_corrections(X, E_ce, y, n_instances, n_counterexamples_per_instance, ce_strategy)
        # add to train set
        X = np.vstack([X, X_corrections])
        y = np.hstack([y, y_corrections])
        print(f"Train set was augmented: X.size= {len(X)}, y.size= {len(y)}")
        one = np.ones(60000, dtype=int)  # first 60000 are non CE
        zero = np.zeros(60000, dtype=int) # next 60000 are CE
        mask = np.append(one, zero) # 0 = CE, 1 = non CE

        E_hint = np.append(E_hint, E_ce)
        E = np.append(E, E_ce)

    if n_expl is not None:
        not_used_flags = np.zeros((E.shape[0] - n_expl), dtype=np.int64)
        used_flags = np.ones(n_expl, dtype=np.int64)
        flags = np.concatenate((used_flags, not_used_flags), axis=0)

    if not flatten:  # if input for a conv net sets should not be flat
        n_samples = X.shape[0]

        X = X.reshape((n_samples, 1, 28, 28))
        Xt = Xt.reshape((10000, 1, 28, 28))
        E = E.reshape(n_samples, 1, 28, 28)
        E_hint = E_hint.reshape(n_samples, 1, 28, 28)
        Et = Et.reshape(10000, 1, 28, 28)

    if device == 'cpu':
        X, y, E, E_hint = torch.from_numpy(X).type(torch.FloatTensor), \
                          torch.from_numpy(y).type(torch.LongTensor), \
                          torch.from_numpy(E).type(torch.LongTensor), \
                          torch.from_numpy(E_hint).type(torch.FloatTensor)

        mask = torch.from_numpy(mask).type(torch.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.FloatTensor), \
                     torch.from_numpy(yt).type(torch.LongTensor), \
                     torch.from_numpy(Et).type(torch.LongTensor)

    else:
        X, y, E, E_hint = torch.from_numpy(X).type(torch.cuda.FloatTensor), \
                          torch.from_numpy(y).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E).type(torch.cuda.LongTensor), \
                          torch.from_numpy(E_hint).type(torch.cuda.FloatTensor)

        mask = torch.from_numpy(mask).type(torch.cuda.LongTensor)

        if n_expl is not None:
            flags = torch.from_numpy(flags).type(torch.cuda.LongTensor)

        Xt, yt, Et = torch.from_numpy(Xt).type(torch.cuda.FloatTensor), \
                     torch.from_numpy(yt).type(torch.cuda.LongTensor), \
                     torch.from_numpy(Et).type(torch.cuda.LongTensor)


    if n_expl is not None:
        train, test = TensorDataset(X, y, E, E_hint, mask, flags), TensorDataset(Xt, yt, Et)
    else:
        train, test = TensorDataset(X, y, E, E_hint, mask), TensorDataset(Xt, yt, Et)

    # print(f"Train set: {train.shape}")
    # print(f"Test set: {test.shape}")
    return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)

def isic_2019(batch_size=16, train_shuffle=True, number_nc=None, number_c=None,\
        ce_augment=False, most_k_informative_img=None, \
        informative_indices_filename='output_wr_metric/informative_score_indices_train_set_most_to_least.npy'):
    """
    Load ISIC Skin Cancer 2019 dataset. 
    Return train and test set Dataloaders.

    Args:
        batch_size: specifies the batch size.
        train_shuffle: sets the pytorch Dataloader attribute 'shuffle' which 
            'have the data reshuffled at every epoch'.
        number_c: limit the number of cancer images.
        number_nc: limit the number of not-cancer images.
        ce_augment: augments the datasets with counterexamples based on the masks (used for CE).
        informative_indices_filename: Filepath to file which stores the indices of the most
            informative instances (use method in explainer.py method to generate file)
    """
    print("\n----------Dataset----------")
    logging.info("----------Dataset----------")
    datapath = "data_store/rawdata/ISIC_2019/ISIC19/"
    start = time.time()
    try:
        print("  Read in data from .h5 files...")
        with h5py.File(datapath + 'not_cancer_imgs.h5', 'r') as hf:
            if number_nc is not None:
                not_cancer_imgs = hf['not_cancer_imgs'][:number_nc]
            else:
                not_cancer_imgs = hf['not_cancer_imgs'][:]
        with h5py.File(datapath + 'not_cancer_masks.h5', 'r') as hf:
            if number_nc is not None:
                not_cancer_masks = hf['not_cancer_masks'][:number_nc]
            else:
                not_cancer_masks = hf['not_cancer_masks'][:]
        with h5py.File(datapath + 'not_cancer_flags.h5', 'r') as hf:
            # indicating wether an instances have a seg mask (1, else 0)
            if number_nc is not None:
                not_cancer_flags = hf['not_cancer_flags'][:number_nc]
            else:
                not_cancer_flags = hf['not_cancer_flags'][:]
        with h5py.File(datapath + 'cancer_imgs.h5', 'r') as hf:
            if number_c is not None:
                cancer_imgs = hf['cancer_imgs'][:number_c]
            else:
                cancer_imgs = hf['cancer_imgs'][:]
    except:
        raise RuntimeError("No isic .h5 files found. Please run the setup at setup_isic.py file!")

    end = time.time()
    elap = int(end - start)
    print(f"  --> Read in finished: Took {elap} sec!")
    logging.info(f"  --> Read in finished: Took {elap} sec!")

    # generate labels: cancer=1; no_cancer=0 
    cancer_targets = np.ones((cancer_imgs.shape[0])).astype(np.int64)
    not_cancer_targets = np.zeros((not_cancer_imgs.shape[0])).astype(np.int64)
    cancer_flags = np.zeros_like(cancer_targets)

    # Generate datasets
    print("  Building datasets...")
    logging.info("  Building datasets...")
    start = time.time()
    not_cancer_dataset = TensorDataset(torch.from_numpy(not_cancer_imgs).float(), \
        torch.from_numpy(not_cancer_targets), torch.from_numpy(not_cancer_masks).float(), \
        torch.from_numpy(not_cancer_flags))
    cancer_dataset = TensorDataset(torch.from_numpy(cancer_imgs).float(), \
        torch.from_numpy(cancer_targets), \
        torch.from_numpy(np.zeros((len(cancer_imgs), 1, 299, 299))).float(),\
        torch.from_numpy(cancer_flags))

    del cancer_imgs, not_cancer_imgs, not_cancer_masks, not_cancer_flags, cancer_targets,\
        not_cancer_targets, cancer_flags
    # Build Datasets
    complete_dataset = ConcatDataset((not_cancer_dataset, cancer_dataset))
    length_complete_dataset = len(complete_dataset)

    # Build train, val and test set.
    num_total = len(complete_dataset)
    num_train = int(0.8 * num_total)
    num_test = num_total - num_train

    train_dataset, test_dataset_ = torch.utils.data.random_split(complete_dataset, \
        [num_train, num_test], generator=torch.Generator().manual_seed(0))

    test_dataset_no_patches = torch.utils.data.Subset(complete_dataset, \
        [idx for idx in test_dataset_.indices if complete_dataset[idx][3] == 0])
    # test with not_cancer images all containing a patch    
    test_dataset = torch.utils.data.Subset(complete_dataset, \
        [idx for idx in test_dataset_.indices if complete_dataset[idx][3] == 1 \
            or complete_dataset[idx][1] == 1])

    if most_k_informative_img is not None:
        with open(informative_indices_filename, 'rb') as f:
            most_informative_ind = np.load(f)[:most_k_informative_img]
        # pytorch TensorDataset does not support assignments to update values so we have to
        # create a new TensorDataset which is equal unless the updatet flags
        imgs, labels, masks, flags = [], [], [], []
        for i, data in enumerate(train_dataset):
            imgs.append(data[0].unsqueeze(0))
            labels.append(data[1].item())
            masks.append(data[2].unsqueeze(0))
            if i in most_informative_ind and data[3] == 1:
                flags.append(1)
            else:
                flags.append(0)

        del train_dataset
        train_dataset = TensorDataset(torch.cat(imgs, 0), torch.Tensor(labels).type(torch.LongTensor), \
            torch.cat(masks, 0), torch.Tensor(flags).type(torch.LongTensor))

        print(f"  MOST {most_k_informative_img} informative images with patches")
        print(f"  --> Train patch dist should have 1 -> {most_k_informative_img}")

    # for CE augmentation
    if ce_augment:
        print(f"  CE Augmentation: Add counterexamples to train set...")
        number = 0
        img_ce, labels_ce, masks_ce, flags_ce = [], [], [], []
        for idx in train_dataset.indices:
            if complete_dataset[idx][3] == 1:
                img_c, label_c, mask_c, flag_c = ce.correct_one_isic(complete_dataset[idx][0], \
                    complete_dataset[idx][1], complete_dataset[idx][2])
                img_ce.append(img_c.unsqueeze(0))
                labels_ce.append(label_c)
                masks_ce.append(mask_c.unsqueeze(0))
                flags_ce.append(flag_c)
                number += 1
        ce_dataset = TensorDataset(torch.cat(img_ce, 0), torch.Tensor(labels_ce).type(torch.LongTensor), torch.cat(masks_ce, 0), torch.cat(flags_ce))
        print(f"  TRAIN ONLY CE SIZE: {len(ce_dataset)}")
        print(f"  TRAIN SET BEFORE CE: {len(train_dataset)}")
        length_before_ce_augmentation = len(train_dataset)
        train_dataset = ConcatDataset((train_dataset, ce_dataset))
        if len(ce_dataset) + length_before_ce_augmentation != len(train_dataset):
            RuntimeWarning(f"Length of trainset before and after ce augmentation incorrect!")
        length_complete_dataset += len(ce_dataset)

    # Calculate ratio between cancerous and not_cancerous for the weighted loss in training

    cancer_ratio = len(cancer_dataset)/ length_complete_dataset
    not_cancer_ratio = 1 - cancer_ratio
    cancer_weight = 1 / cancer_ratio
    not_cancer_weight = 1 / not_cancer_ratio
    weights = np.asarray([not_cancer_weight, cancer_weight])
    weights /= weights.sum()
    weights = torch.tensor(weights).float()

    datasets = {'train': train_dataset, 'test': test_dataset, 'test_no_patches': test_dataset_no_patches}
    # tt = ConcatDataset((train_dataset, test_dataset_no_patches))

    print("  Sizes of datasets:")
    print(f"  TRAIN: {len(train_dataset)}, TEST: {len(test_dataset)}, TEST_NO_PATCHES: {len(test_dataset_no_patches)}")

    ######################### only for checking the data distribution in trainset
    train_classes = [x[1].item() for x in train_dataset]
    train_patch_dis = [x[3].item() for x in train_dataset]

    #train_classes = [complete_dataset[idx][1].item() for idx in train_dataset.indices]
    #train_patch_dis = [complete_dataset[idx][3].item() for idx in train_dataset.indices]
    print(f"  TRAIN class dist: {Counter(train_classes)}")
    print(f"  TRAIN patch dist: {Counter(train_patch_dis)}") # 0 -> no patch, 1-> patch
    test_classes = [complete_dataset[idx][1].item() for idx in test_dataset.indices]
    print(f"  TEST class dist: {Counter(test_classes)}")
    test_classes_no_patches = [complete_dataset[idx][1].item() for idx in test_dataset_no_patches.indices]
    print(f"  TEST_NO_PATCHES class dist: {Counter(test_classes_no_patches)}")
    print(f"  Loss weights: {str(weights)}")

    logging.info("  Sizes of datasets:")
    logging.info(f"  TRAIN: {len(train_dataset)}, TEST: {len(test_dataset)}, TEST_NO_PATCHES: {len(test_dataset_no_patches)}")
    logging.info(f"  TRAIN class dist: {Counter(train_classes)}")
    logging.info(f"  TRAIN patch dist: {Counter(train_patch_dis)}")
    logging.info(f"  TEST class dist: {Counter(test_classes)}")
    logging.info(f"  TEST_NO_PATCHES class dist: {Counter(test_classes_no_patches)}")
    logging.info(f"  Loss weights: {str(weights)}")


    dataloaders = {}
    dataloaders['train'] = DataLoader(datasets['train'], batch_size=batch_size, \
        shuffle=train_shuffle)
    dataloaders['test'] = DataLoader(datasets['test'], batch_size=batch_size, \
        shuffle=False)
    dataloaders['test_no_patches'] = DataLoader(datasets['test_no_patches'], batch_size=batch_size, \
        shuffle=False)

    end = time.time()
    elap = int(end - start)

    print(f"  --> Build finished: Took {elap} sec!")
    print("--------Dataset Done--------\n")
    logging.info(f"  --> Build finished: Took {elap} sec!")
    logging.info("--------Dataset Done--------\n")
    return dataloaders, weights

def isic_2019_hint(batch_size=16, train_shuffle=True, number_nc=None, number_c=None,\
        all_hint=False, invert_seg_hint_masks=False):
    """
    Load ISIC Skin Cancer 2019 dataset for the HINT. 
    Return train, validation and test Dataloaders.

    (Used for HINT method and Reward vs. Penalize experiment)

    Args:
        batch_size: specifies the batch size.
        train_shuffle: sets the pytorch Dataloader attribute 'shuffle' which 
            'have the data reshuffled at every epoch'.
        number_c: limit the number of cancer images.
        number_nc: limit the number of not-cancer images.
        all_hint: set to True if all of the available right reason mask should be used.
            If False then only instances with a patch get the hint masks.
        invert_seg_hint_masks: if True then masks penalize whole background. 
    """
    print("\n----------Dataset (HINT)------")
    logging.info("----------Dataset----------")
    datapath = "data_store/rawdata/ISIC_2019/ISIC19/"
    start = time.time()
    try:
        print("  Read in data from .h5 files...")
        with h5py.File(datapath + 'not_cancer_imgs.h5', 'r') as hf:
            if number_nc is not None:
                not_cancer_imgs = hf['not_cancer_imgs'][:number_nc]
            else:
                not_cancer_imgs = hf['not_cancer_imgs'][:]
        with h5py.File(datapath + 'not_cancer_masks_hint.h5', 'r') as hf:
            if number_nc is not None:
                not_cancer_masks_hint = hf['not_cancer_masks_hint'][:number_nc]
            else:
                not_cancer_masks_hint = hf['not_cancer_masks_hint'][:]

            if invert_seg_hint_masks:
                not_cancer_masks_hint = (~not_cancer_masks_hint.astype(np.bool)).astype(np.float)
        with h5py.File(datapath + 'not_cancer_flags.h5', 'r') as hf:
            # indicating wether an instances have a seg mask (1, else 0)
            if number_nc is not None:
                not_cancer_flags = hf['not_cancer_flags'][:number_nc]
            else:
                not_cancer_flags = hf['not_cancer_flags'][:]
        with h5py.File(datapath + 'not_cancer_flags_hint.h5', 'r') as hf:
            # indicating wether an instances have a seg hint mask (1, else 0)
            if number_nc is not None:
                not_cancer_flags_hint = hf['not_cancer_flags_hint'][:number_nc]
            else:
                not_cancer_flags_hint = hf['not_cancer_flags_hint'][:]
        with h5py.File(datapath + 'cancer_imgs.h5', 'r') as hf:
            if number_c is not None:
                cancer_imgs = hf['cancer_imgs'][:number_c]
            else:
                cancer_imgs = hf['cancer_imgs'][:]
        if all_hint:
            with h5py.File(datapath + 'cancer_masks_hint.h5', 'r') as hf:
                if number_c is not None:
                    cancer_masks_hint = hf['cancer_masks_hint'][:number_c]
                else:
                    cancer_masks_hint = hf['cancer_masks_hint'][:]
                if invert_seg_hint_masks:
                    cancer_masks_hint = (~cancer_masks_hint.astype(np.bool)).astype(np.float)
            with h5py.File(datapath + 'cancer_flags_hint.h5', 'r') as hf:
                if number_c is not None:
                    cancer_flags_hint = hf['cancer_flags_hint'][:number_c]
                else:
                    cancer_flags_hint = hf['cancer_flags_hint'][:]
    except:
        raise RuntimeError("No isic .h5 files found. Please run the setup at setup_isic.py file!")

    end = time.time()
    elap = int(end - start)
    print(f"  --> Read in finished: Took {elap} sec!")
    logging.info(f"  --> Read in finished: Took {elap} sec!")

    # generate labels: cancer=1; no_cancer=0 
    cancer_targets = np.ones((cancer_imgs.shape[0])).astype(np.int64)
    not_cancer_targets = np.zeros((not_cancer_imgs.shape[0])).astype(np.int64)

    if not all_hint:
        del cancer_flags_hint, cancer_masks_hint
        cancer_flags_hint = np.zeros_like(cancer_targets)
        cancer_masks_hint = np.zeros((len(cancer_imgs), 1, 299, 299))

        # adapt hint_flags for not_cancer only having only ones if img has patch 
        # according to orginal flags
        if len(not_cancer_flags) != len(not_cancer_flags_hint):
            raise RuntimeWarning(f"flags and hint_flags do not match: Flags= {len(not_cancer_flags)} vs. flags hint= {len(not_cancer_flags_hint)}")

        print(f"  HINT FLAGS ADAPTION: Before sum={np.sum(not_cancer_flags_hint)}")
        not_cancer_flags_hint = not_cancer_flags_hint & not_cancer_flags
        print(f"  HINT FLAGS ADAPTION: After  sum={np.sum(not_cancer_flags_hint)}")

    # Generate datasets
    print("  Building datasets...")
    logging.info("  Building datasets...")
    start = time.time()
    not_cancer_dataset = TensorDataset(torch.from_numpy(not_cancer_imgs).float(), \
        torch.from_numpy(not_cancer_targets), torch.from_numpy(not_cancer_masks_hint).float(), \
        torch.from_numpy(not_cancer_flags_hint), torch.from_numpy(not_cancer_flags))
    cancer_dataset = TensorDataset(torch.from_numpy(cancer_imgs).float(), \
        torch.from_numpy(cancer_targets), \
        torch.from_numpy(cancer_masks_hint).float(),\
        torch.from_numpy(cancer_flags_hint), torch.from_numpy(np.zeros_like(cancer_targets)))

    del cancer_imgs, not_cancer_imgs, not_cancer_masks_hint, not_cancer_flags, cancer_targets,\
        not_cancer_targets, cancer_flags_hint
    # Build Datasets
    complete_dataset = ConcatDataset((not_cancer_dataset, cancer_dataset))

    length_complete_dataset = len(complete_dataset)

    # Build train, val and test set.
    num_total = len(complete_dataset)
    num_train = int(0.8 * num_total)
    num_test = num_total - num_train

    train_dataset, test_dataset_ = torch.utils.data.random_split(complete_dataset, \
        [num_train, num_test], generator=torch.Generator().manual_seed(0))

    test_dataset_no_patches = torch.utils.data.Subset(complete_dataset, \
        [idx for idx in test_dataset_.indices if complete_dataset[idx][4] == 0])
    # test with not_cancer images all containing a patch    
    test_dataset = torch.utils.data.Subset(complete_dataset, \
        [idx for idx in test_dataset_.indices if complete_dataset[idx][4] == 1 \
            or complete_dataset[idx][1] == 1])

    # Calculate ratio between cancerous and not_cancerous for the weighted loss in training

    cancer_ratio = len(cancer_dataset)/ length_complete_dataset
    not_cancer_ratio = 1 - cancer_ratio
    cancer_weight = 1 / cancer_ratio
    not_cancer_weight = 1 / not_cancer_ratio
    weights = np.asarray([not_cancer_weight, cancer_weight])
    weights /= weights.sum()
    weights = torch.tensor(weights).float()

    datasets = {'train': train_dataset, 'test': test_dataset, 'test_no_patches': test_dataset_no_patches}
    # tt = ConcatDataset((train_dataset, test_dataset_no_patches))

    print("  Sizes of datasets:")
    print(f"  TRAIN: {len(train_dataset)}, TEST: {len(test_dataset)}, TEST_NO_PATCHES: {len(test_dataset_no_patches)}")

    ######################### only for checking the data distribution in trainset
    train_classes = [x[1].item() for x in train_dataset]
    train_patch_dis = [x[3].item() for x in train_dataset]

    #train_classes = [complete_dataset[idx][1].item() for idx in train_dataset.indices]
    #train_patch_dis = [complete_dataset[idx][3].item() for idx in train_dataset.indices]
    print(f"  TRAIN class dist: {Counter(train_classes)}")
    print(f"  TRAIN mask dist: {Counter(train_patch_dis)}") # 0 -> no patch, 1-> patch
    test_classes = [complete_dataset[idx][1].item() for idx in test_dataset.indices]
    print(f"  TEST class dist: {Counter(test_classes)}")
    test_masks = [complete_dataset[idx][3].item() for idx in test_dataset.indices]
    print(f"  TEST mask dist: {Counter(test_masks)}")
    test_classes_no_patches = [complete_dataset[idx][1].item() for idx in test_dataset_no_patches.indices]
    print(f"  TEST_NO_PATCHES class dist: {Counter(test_classes_no_patches)}")
    test_masks_no_patches = [complete_dataset[idx][3].item() for idx in test_dataset_no_patches.indices]
    print(f"  TEST_NO_PATCHES mask dist: {Counter(test_masks_no_patches)}")
    print(f"  Loss weights: {str(weights)}")

    logging.info("  Sizes of datasets:")
    logging.info(f"  TRAIN: {len(train_dataset)}, TEST: {len(test_dataset)}, TEST_NO_PATCHES: {len(test_dataset_no_patches)}")
    logging.info(f"  TRAIN class dist: {Counter(train_classes)}")
    logging.info(f"  TRAIN mask dist: {Counter(train_patch_dis)}")
    logging.info(f"  TEST class dist: {Counter(test_classes)}")
    logging.info(f"  TEST masks dist: {Counter(test_masks)}")
    logging.info(f"  TEST_NO_PATCHES class dist: {Counter(test_classes_no_patches)}")
    logging.info(f"  TEST NO PATCHES masks dist: {Counter(test_masks_no_patches)}")
    logging.info(f"  Loss weights: {str(weights)}")


    dataloaders = {}
    dataloaders['train'] = DataLoader(datasets['train'], batch_size=batch_size, \
        shuffle=train_shuffle)
    dataloaders['test'] = DataLoader(datasets['test'], batch_size=batch_size, \
        shuffle=False)
    dataloaders['test_no_patches'] = DataLoader(datasets['test_no_patches'], batch_size=batch_size, \
        shuffle=False)

    end = time.time()
    elap = int(end - start)

    print(f"  --> Build finished: Took {elap} sec!")
    print("--------Dataset Done--------\n")
    logging.info(f"  --> Build finished: Took {elap} sec!")
    logging.info("--------Dataset Done--------\n")
    return dataloaders, weights


def decoy_mnist_all_revised(fmnist=False, batch_size=256, device='cuda', train_shuffle=False, \
                    test_shuffle=False,feedback=None, n_expl=None, flatten=False, \
                    n_instances=-1, n_counterexamples_per_instance=1, generate_counterexamples=False, counterexample_strategy='random', reduced_training_size=None):
   
    if fmnist:
        _, X, y, E_pnlt, E_rwrd, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/fashionMnist/decoy-fmnist.npz', fmnist=True)
    else:
        _, X, y, E_pnlt, E_rwrd, _, Xt, yt, Et, _ = load_decoy_mnist.generate_dataset( \
            cachefile='data_store/rawdata/MNIST/decoy-mnist.npz')


    E_ce = E_pnlt

    if feedback is not None:
        n, d = E_pnlt.shape[0], E_pnlt.shape[1]
        if feedback == 'random':
            # set explanation feedback masks to random masks
            E_pnlt = np.random.randint(2, size=n * d).reshape(n, d)
            E_ce = np.random.randint(2, size=n*d).reshape(n, d).astype(bool)
            E_rwrd = np.random.randint(2, size=n * d).reshape(n, d)

        elif feedback == 'adversarial':
            # set explanation feedback masks to all ones
            E_pnlt = np.ones((n, d), dtype=np.int64)
            E_ce = np.ones((n,d), dtype=np.int64).astype(bool)
            E_rwrd = np.ones((n, d), dtype=np.int64)

        elif feedback == 'incomplete':
            # delete top half of the feedback mask (4x4 squares -> 2x4 squares)
            for i, e in enumerate(E_pnlt):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E_pnlt[i] = e

            for i, e in enumerate(E_ce):
                true_indexes = np.where(e == True)[0]
                first_half_true_indexes = true_indexes[:8]
                np.put(e, first_half_true_indexes, np.array(False))
                E_ce[i] = e.astype(bool)

            for i, e in enumerate(E_rwrd):
                true_indexes = np.where(e == True)[0]
                half_mask_number = int(true_indexes.size / 2)
                first_half_true_indexes = true_indexes[:half_mask_number]
                np.put(e, first_half_true_indexes, np.array(False))
                E_rwrd[i] = e

        elif feedback == 'wrong':
            # add a 5 x 3 rectangle in the top middle or bottom middle on the border
            # rectangle is placed on the opposite side of the the confounder square
            if not fmnist:
                for i, e in enumerate(E_pnlt):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_pnlt[i] = cur

                for i, e in enumerate(E_ce):
                    det = np.where(e == True)[0]
                    if det.item(0) in [x for x in range(0, 28)]:  # top row
                        ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                    else:  # bottom row
                        ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_ce[i] = cur

                E_rwrd = np.copy(E_pnlt)

            if fmnist:
                for i, e in enumerate(E_pnlt):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_pnlt[i] = cur

                for i, e in enumerate(E_ce):
                    det = np.where(e == True)[0]
                    if y[i] in [0, 1, 2, 3, 4, 6, 8]:  # rectangle left right
                        if det.item(0) in [0, 672]:  # left
                            ind = [306, 307, 334, 335, 362, 363, 390, 391, 418, 419, 446, 447, 474, 475, 502, 503]
                        else:  # right
                            ind = [280, 281, 308, 309, 336, 337, 364, 365, 392, 393, 420, 421, 448, 449, 476, 477]
                    else:  # rectangle bottom top row
                        if det.item(0) in [0, 24]:  # top
                            ind = [738, 739, 740, 741, 742, 743, 744, 745, 766, 767, 768, 769, 770, 771, 772, 773]
                        else:  # bottom
                            ind = [10, 11, 12, 13, 14, 15, 16, 17, 38, 39, 40, 41, 42, 43, 44, 45]
                    cur = np.zeros(d, dtype=np.int64)
                    cur[ind] = 1
                    E_ce[i] = cur

                E_rwrd = np.copy(E_pnlt)

    # initialize mask all non-ce
    counterexample_mask = np.zeros(60_000, dtype=int) # 0 = non CE, 1 = CE
    if generate_counterexamples:
        X_corrections, y_corrections = ce.get_corrections(X, E_ce, y, n_instances, n_counterexamples_per_instance, counterexample_strategy)

        # add counterexamples to train set
        X = np.vstack([X, X_corrections])
        y = np.hstack([y, y_corrections])

        logging.info(f"Train set was augmented with counterexamples: X.size= {len(X)}, y.size= {len(y)}")
        
        # first 60000 are non CE, next 60000 are CE
        counterexample_mask = np.append(np.zeros(60_000, dtype=int), np.ones(60_000, dtype=int)) # 0 = non CE, 1 = CE

        # just fill-up explanations with the non-ce values (won't be used)
        E_rwrd = np.append(E_rwrd, E_ce)
        E_pnlt = np.append(E_pnlt, E_ce)

    if n_expl:
        not_used_flags = np.zeros((E_pnlt.shape[0] - n_expl), dtype=np.int64)
        used_flags = np.ones(n_expl, dtype=np.int64)
        flags = np.concatenate((used_flags, not_used_flags), axis=0)

    if not flatten:  # if input for a conv net sets should not be flat
        n_samples = X.shape[0]

        X = X.reshape((n_samples, 1, 28, 28))
        Xt = Xt.reshape((10000, 1, 28, 28))
        E_pnlt = E_pnlt.reshape(n_samples, 1, 28, 28)
        E_rwrd = E_rwrd.reshape(n_samples, 1, 28, 28)
        Et = Et.reshape(10000, 1, 28, 28)

    # convert to torch tensors
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).long().to(device)
    E_pnlt = torch.from_numpy(E_pnlt).long().to(device)
    E_rwrd = torch.from_numpy(E_rwrd).float().to(device)

    Xt = torch.from_numpy(Xt).float().to(device)
    yt = torch.from_numpy(yt).long().to(device)
    Et = torch.from_numpy(Et).long().to(device)
    
    counterexample_mask = torch.from_numpy(counterexample_mask).bool().to(device)

    if n_expl:
        flags = torch.from_numpy(flags).to(device)


    logging.debug(f"X.shape={X.shape}, Xt.shape={Xt.shape}, y.shape={y.shape}, yt.shape={yt.shape}")

    if n_expl:
        train, test = TensorDataset(X, y, E_pnlt, E_rwrd, counterexample_mask, flags), TensorDataset(Xt, yt, Et)
    else:
        train, test = TensorDataset(X, y, E_pnlt, E_rwrd, counterexample_mask), TensorDataset(Xt, yt, Et)

    if reduced_training_size:
        # train contains 50% counterexamples and 50% non-counterexamples
        # when subsetting to N instances we want to get N/2 CEs and N/2 non-CEs and we collect them starting from the middle index
        train_start, train_end = int((len(train) - reduced_training_size) / 2), int((len(train) + reduced_training_size) / 2)
        train = Subset(train, range(train_start, train_end))
        logging.warn(f"running with reduced_training_size={reduced_training_size}! instances={train_start}:{train_end}")

    return DataLoader(train, batch_size, train_shuffle), DataLoader(test, batch_size, test_shuffle)
