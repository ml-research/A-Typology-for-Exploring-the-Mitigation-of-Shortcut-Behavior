# from https://github.com/dtak/rrr/blob/master/rrr/decoy_mnist.py
from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()
import os
import gzip
import struct
import array
import numpy as np
from urllib.request import urlretrieve

def download_mnist(datadir, fmnist=False):
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    if fmnist:
        base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    else:
        base_url = 'http://yann.lecun.com/exdb/mnist/'
    
    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)
            
    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)
            
    for filename in ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',\
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']:
        if not os.path.exists(os.path.join(datadir, filename)):
            print(f"Downloading raw files from {base_url + filename}")
            urlretrieve(base_url + filename, os.path.join(datadir, filename))
            
    train_images = parse_images(os.path.join(datadir, 'train-images-idx3-ubyte.gz'))
    train_labels = parse_labels(os.path.join(datadir, 'train-labels-idx1-ubyte.gz'))
    test_images = parse_images(os.path.join(datadir, 't10k-images-idx3-ubyte.gz'))
    test_labels = parse_labels(os.path.join(datadir, 't10k-labels-idx1-ubyte.gz'))
    
    return train_images, train_labels, test_images, test_labels

def Bern(p):
    return np.random.rand() < p

def augment(image, digit, randomize=False, mult=25, all_digits=range(10)):
    if randomize:
        return augment(image, np.random.choice(all_digits))

    img = image.copy()
    expl = np.zeros_like(img)

    fwd = [0,1,2,3]
    rev = [-1,-2,-3,-4]
    dir1 = fwd if Bern(0.5) else rev
    dir2 = fwd if Bern(0.5) else rev
    for i in dir1:
        for j in dir2:
            img[i][j] = 255 - mult*digit
            expl[i][j] = 1
    
    expl_hint = image.copy()
    expl_hint = np.ma.masked_where(expl_hint > 1, expl_hint)
    expl_hint = np.ma.getmask(expl_hint)

    return img.ravel(), expl.astype(bool).ravel(), expl_hint.ravel()

def _generate_dataset(datadir, fmnist):
    X_raw, y, Xt_raw, yt = download_mnist(datadir, fmnist)
    all_digits = list(set(y))
    X = []
    E = []
    E_hint = []
    Xt = []
    Et = []
    Et_hint = []
    
    for image, digit in zip(X_raw, y):
        x, e, e_hint = augment(image, digit, all_digits=all_digits)
        X.append(x)
        E.append(e)
        E_hint.append(e_hint)
        
    for image, digit in zip(Xt_raw, yt):
        x, e, e_hint = augment(image, digit, all_digits=all_digits, randomize=True)
        Xt.append(x)
        Et.append(e)
        Et_hint.append(e_hint)
        
    X = np.array(X)
    E = np.array(E)
    E_hint = np.array(E_hint)
    Xt = np.array(Xt)
    Et = np.array(Et)
    Et_hint = np.array(Et_hint)
    Xr = np.array([x.ravel() for x in X_raw])
    Xtr = np.array([x.ravel() for x in Xt_raw])
    
    return Xr, X, y, E, E_hint, Xtr, Xt, yt, Et, Et_hint

def generate_dataset(cachefile='data_store/rawdata/MNIST/decoy-mnist.npz', fmnist=False):
    if cachefile and os.path.exists(cachefile):
        print("Loading dataset from existing file!")
        cache = np.load(cachefile)
        data = tuple([cache[f] for f in sorted(cache.files)])
    else:
        print("Generating dataset...")
        data = _generate_dataset(os.path.dirname(cachefile) + '/raw', fmnist)
        if cachefile:
            np.savez(cachefile, *data)
    return data
