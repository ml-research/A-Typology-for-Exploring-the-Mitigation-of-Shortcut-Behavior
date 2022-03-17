"""Collection of utility functions."""
import random
import os
import shutil

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from captum.attr import visualization as viz
import pandas as pd

def seed_all(seed=60):
    """Seed all for reproducability."""
    print(f"[Using Seed= {seed}]")
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def empty_log_run_model_store_folders():
    """Remove all files in the output folders."""
    dirs = ['logs/', 'runs/', 'learner/model_store/']
    for dir in dirs:
        for files in os.listdir(dir):
            path = os.path.join(dir, files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)      

def norm_saliencies(saliencies):
    """
    Normalize tensor across first dimension according to formula 
    t(i) = (i - min_t) /(max_t - min_t).
    Note: uses for loop -> slow
    
    Args:
        saliencies: tensor of shape (n, c, h, w).
    """
    saliencies_norm = saliencies.clone()
    for i in range(saliencies.shape[0]):
        #max, min = torch.max(saliencies[i]), torch.min(saliencies[i])
        if (torch.max(saliencies[i]) - torch.min(saliencies[i]) != 0.):
            saliencies_norm[i] = (saliencies[i] - torch.min(saliencies[i]))/\
                 (torch.max(saliencies[i]) - torch.min(saliencies[i]))
        #print(saliencies_norm[i])
    return saliencies_norm

def norm_saliencies_fast(A, positive_only=False):
    """
    Normalize tensor to [0,1] across first dimension (for every batch_i) according to formula 
    t(i) = (i_t - min_t) /(max_t + 1e-6).
    Add small constant to prevent zero divison.
    
    Args:
        A: tensor of shape (n, c, h, w).
        positive_only: if True then take only positive values into account and 
            zero out negative values.
    """
    shape = A.shape
    A = A.view(A.size(0), -1)
    if positive_only:
        A[A < 0] = 0.
    A -= A.min(1, keepdim=True)[0]
    A /= (A.max(1, keepdim=True)[0] + 1e-6) # add small constant preventing zero divison
    A = A.view(shape)
    return A

def get_white_viridis_cmap(range_start=96, range_end=196):
    """Colormap for the heatmaps."""

    viridis = cm.get_cmap('viridis', 256)
    viridis_colors = viridis.colors.copy()

    viridis_white = viridis_colors[range_end + 1].copy()

    viridis_colors[0:range_start, 0] = np.ones(range_start)
    viridis_colors[0:range_start, 1] = np.ones(range_start)
    viridis_colors[0:range_start, 2] = np.ones(range_start)
    viridis_colors[0:range_start, 3] = np.ones(range_start)

    viridis_colors[range_start:range_end, 0] = np.linspace(1, viridis_white[0], range_end - range_start)
    viridis_colors[range_start:range_end, 1] = np.linspace(1, viridis_white[1], range_end - range_start)
    viridis_colors[range_start:range_end, 2] = np.linspace(1, viridis_white[2], range_end - range_start)
    viridis_colors[range_start:range_end, 3] = np.linspace(1, viridis_white[3], range_end - range_start)

    newcmp = ListedColormap(viridis_colors)
    return newcmp

def get_last_conv_layer(model):
    """Return last conv layer of a pytorch model."""
    index = None
    modules = [module for module in model.modules() if not isinstance(module, nn.Sequential)][1:]
    for i, module in enumerate(modules):
        if 'Conv' in str(module):
            index = i
    if index is None:
        raise Exception("Model has no conv layer!")
    return modules[index]

def load_first_batch(dataloader):
    """
    Warning: If dataloader does have attribute shuffle=True then function 
        does not return the excat same batch if function called twice while 
        the dataloader is not reloaded outside this function.  
    """
    for data in dataloader:
        X, labels = data[0], data[1]
        # if len(data) == 3:
        #     X, labels, _ = data
        #     return X, labels
        # else:
        #     X, labels = data
        return X, labels

def show_explanation_single(img, mask, method='heat_map', sign='positive', save_name=None):
    fig, ax = plt.subplots()
    viz.visualize_image_attr(mask[0], img[0], method=method, cmap=get_white_viridis_cmap(0,128), sign=sign, \
                             show_colorbar=True, use_pyplot=False, plt_fig_axis=(fig, ax), alpha_overlay=1.0)
    
    if save_name is not None:
        name_ = 'output_images/' + save_name + '.png'
        plt.savefig(name_, bbox_inches='tight')
        plt.close()        
    else:    
        plt.show()


def show_explanation_overlay_grid_captum(imgs, masks, titles, figsize=(12,6), \
    n_col=4, method='blended_heat_map', sign='positive', fig_header='?', save_name=None):
    """
    Show heatmaps.
    
    Args:
        imgs: list of numpy arrays.
        masks: list of numpy arrays.
        titles: list of strings.
        figsize: sets the size of the figure.
        n_col: sets the number of columns in the plot.
        method: specify the method with wihich the plot is generated (same as in captum library).
            'blended_heat_map' -> lay heatmap on top of image.
            'heat_map'         -> only plot heatmap.
            'original_image'   -> only display original image.
            'masked_image'  -> Mask image (pixel-wise multiply) by normalized attribution values.
            'alpha_scaling' -> Sets alpha channel of each pixel to be equal 
                to normalized attribution value.
        sign: chosen sign of attributions to visualize. Supported options are:
            'positive' -> displays only positive pixel attributions.
            'absolute_value' -> displays absolute value of attributions.
            'negative' -> displays only negative pixel attributions.
        fig_header: haeder of the generated plot.
    """

    n_row = int(np.ceil(len(imgs) / n_col))
    fig, axs = plt.subplots(n_row, n_col, figsize=figsize)
    fig.suptitle('Explanation generated with ' + fig_header, fontsize=13)
    fig.subplots_adjust(hspace=0.4)

    axs = axs.flatten()
    for img, mask, title, ax in zip(imgs, masks, titles, axs):
        viz.visualize_image_attr(mask, img, method=method, sign=sign,\
            show_colorbar=True, title=title, cmap=get_white_viridis_cmap(0,128), alpha_overlay=0.7, \
                use_pyplot=False, plt_fig_axis=(fig, ax))
    
    if save_name is not None:
        name_ = 'output_images/' + save_name + '.png'
        plt.savefig(name_, bbox_inches='tight')
        plt.close()
    else:    
        plt.show()

def show_explanation_overlay_grid_captum_2(imgs, masks, titles, figsize=(6,7), n_col=2, method='heat_map',\
    sign='positive', fig_header='?', save_name=None):
    """Show image and heatmaps next to each other."""

    #n_row = int(np.ceil(len(imgs) / n_col)*2)
    n_row = len(imgs)
    fig, axs = plt.subplots(n_row, n_col, figsize=figsize)
    fig.suptitle('Explanation generated with ' + fig_header, fontsize=13)
    fig.subplots_adjust(hspace=0.4)

    axs = axs.flatten()
    is_img = 0
    img_counter = 0
    mask_counter = 0
    for ax in axs:
        if is_img == 0:
            if img_counter >= len(imgs):
                break  
            ax.imshow(imgs[img_counter], cmap='gray', interpolation='none')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            is_img = 1
            img_counter += 1
        else:
            viz.visualize_image_attr(masks[mask_counter], imgs[mask_counter], method=method, sign=sign,\
                show_colorbar=True, title=titles[mask_counter], cmap=get_white_viridis_cmap(0,128), \
                    use_pyplot=False, plt_fig_axis=(fig, ax)) 
            is_img = 0
            mask_counter += 1

    if save_name is not None:
        name_ = 'output_images/' + save_name + '.png'
        plt.savefig(name_, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def show_img_grid_numpy(imgs, figsize=(10,4), n_col=4, save_name=None):

    n_row = int(np.ceil(len(imgs) / n_col))
    fig, axs = plt.subplots(n_row, n_col, figsize=figsize)
    fig.suptitle('Orginial images', fontsize=13)
    fig.subplots_adjust(hspace=0.2)
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img, cmap='gray', interpolation='none')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if save_name is not None:
        name_ = 'output_images/' + save_name + '.png'
        plt.savefig(name_, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_ra_rr_loss_from_log(filename, x_axis, size):
    df = pd.read_csv('logs/'+filename)
    #Then plot using pandas:
    df.plot(x=x_axis, y=['ra_loss', 'rr_loss'], figsize=size)
    plt.ylabel("number of instances")
    plt.ylabel("ra vs rr loss")
    plt.show()

def plot_tensor_rgb_with_expl(intstance):
    img, label, expl, flag = intstance
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"y={label.item()}, flag={flag.item()}")
    plt.show()
    if flag.item() != 0:
        plt.imshow(expl.permute(1, 2, 0), cmap='gray')
        plt.show()

def plot_images(dataset, index):
    img, label, expl = dataset[index]
    img = img.view([28,28])
    plt.imshow(img.numpy(), cmap='gray')
    plt.show()
    print(label)
    img = expl.view([28,28])
    plt.imshow(img.numpy(), cmap='gray')
    plt.show()

def show_image_tensor(img, size, name_):
    img = img.view(size)
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(img.detach().numpy(), cmap='gray')
    plt.savefig(name_, bbox_inches='tight')
    plt.close()

def show_image_numpy(img, shape=(28,28)):
    img = img.reshape(shape)
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(img, cmap='gray')
    plt.show()

def show_image_numpy_rgb(img):
    plt.figure(1)
    plt.subplot(211)
    plt.imshow(np.transpose(img, (1,2,0)))
    plt.show()

def show_img_expl_grid_mnist(dataloader, size=(12,4)):
    """Shows image grid for mnist torch trainloader with explanations."""
    
    collect_imgs = []
    already = []

    for _, (imgs, labels, expls) in enumerate(dataloader):
        for i in range(list(labels.size())[0]):
            if len(collect_imgs) == 10:
                break
            #print(expls[i])
            number = labels[i].item()
            if number not in already:
                collect_imgs.append((imgs[i].squeeze().cpu().numpy(), number, expls[i].squeeze().cpu().numpy()))
                already.append(number)
    
    collect_imgs.sort(key=lambda x: x[1])
    explanations = [(e, 0, 0) for _,_, e in collect_imgs]
    collect_imgs = collect_imgs + explanations

    fig = plt.figure(figsize=size)
    columns = 10
    rows = 2
    for i in range(1, rows*columns+1):
        img, label, _ = collect_imgs[i-1]
        fig.add_subplot(rows, columns, i)
        if i < 11:
            plt.imshow(img, cmap='gray', interpolation='none')
        else:
            plt.imshow(img, cmap='gray', interpolation='none', vmin=0, vmax=1)
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if i < 11:
            plt.title(f"label {label}")
        else:
            plt.title(f"mask")
    plt.show()

def show_img_grid_mnist(dataloader, size=(10,4)):
    """Shows image grid for mnist torch testloader (no explanations)."""
    
    collect_imgs = []
    already = []

    for _, (imgs, labels) in enumerate(dataloader):
        for i in range(list(labels.size())[0]):
            if len(collect_imgs) == 10:
                break
        
            number = labels[i].item()
            if number not in already:
                collect_imgs.append((imgs[i].squeeze().cpu().numpy(), number))
                already.append(number)
    
    collect_imgs.sort(key=lambda x: x[1])

    fig = plt.figure(figsize=size)
    columns = 5
    rows = 2
    for i in range(1, rows*columns+1):
        img, label = collect_imgs[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray', interpolation='none')
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        plt.title(f"label {label}")
        
    plt.show()
