"""
Functions used for the CE method.
Based on https://www.nature.com/articles/s42256-020-0212-3 
"""
import numpy as np
import torch

def correct_one(x, e, label, n_counterexamples, ce_strategy):
    # expects numpy arrays
    x = x.reshape((28, 28)) # 28 x 28 x [0 ... 255]
    e = e.reshape((28, 28)) # 28 x 28 x {False, True}

    X_counterexamples = []
    for _ in range(n_counterexamples):
        x_counterexample = np.array(x, copy=True)
        if ce_strategy == 'random':
            x_counterexample[e] = np.random.randint(0, 256, size=x[e].shape)
        else:
            print("No other than random CE strategy implemented till yet!")
            raise RuntimeError
        X_counterexamples.append(x_counterexample.ravel())
        
    return X_counterexamples


def get_corrections(X, E, y, n_instances, n_counterexamples, ce_strategy):
    """
    Generates corrections. X=Inputs, E=Explanations
    """
    # expects numpy arrays
    X_counterexamples, y_counterexamples = [], []
    if n_instances == -1:
        for x, e, label in zip(X, E, y):
            temp = correct_one(x, e, label, n_counterexamples, ce_strategy)
            X_counterexamples.extend(temp)
            y_counterexamples.extend([label] * len(temp))
    else:
        for n, (x, e, label) in enumerate(zip(X, E, y)):
            if n == n_instances:
                break

            temp = correct_one(x, e, label, n_counterexamples, ce_strategy)
            X_counterexamples.extend(temp)
            y_counterexamples.extend([label] * len(temp))

    return np.array(X_counterexamples), np.array(y_counterexamples)

def correct_one_isic(img_t, label_t, mask_t, ce_strategy='random'):
    """
    Generate one counterexample for one isic image with a patch.

    Args:
        img_t: image tensor of shape (3, h, w)
        label_t: tensor of label
        mask_t: maks tensor of shape (1, h, w)
        ce_strategy: currently only the randomize strategy is supported
    """
    img_np = img_t.detach().cpu().numpy().copy()
    label = label_t.detach().cpu().item()
    mask_np = np.squeeze(mask_t.detach().cpu().numpy().copy().astype(np.int))
    num = np.sum(mask_np).astype(np.int)

    if ce_strategy == 'random':
        np.place(img_np[0], mask_np, np.random.uniform(0,1, size=num).astype(np.float32))
        np.place(img_np[1], mask_np, np.random.uniform(0,1, size=num).astype(np.float32))
        np.place(img_np[2], mask_np, np.random.uniform(0,1, size=num).astype(np.float32))

    else:
        print("No other than random CE strategy implemented till yet!")
        raise RuntimeError
    
    img_c_t, label_c_t, mask_c_t, flag_c_t = torch.from_numpy(img_np), label, \
        mask_t.detach().clone(), -torch.ones(1).type(torch.int)
    
    return img_c_t, label_c_t, mask_c_t, flag_c_t
