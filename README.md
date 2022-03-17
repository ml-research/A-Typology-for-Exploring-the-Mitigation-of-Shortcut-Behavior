<!-- #region -->
# Programming Framework of XIL Typology
Repository for the paper [A Typology to Explore and Guide Explanatory Interactive Machine Learning](https://arxiv.org/abs/2203.03668). All developed components use the PyTorch library and are evaluated on CV tasks. This repository implements and examines the following XIL methods:

* Right for Right Reason (RRR) (Ross et. al., 2017)
* Right for the Better Reason (RBR) (Shao et. al., 2021)
* Right for the Right Reason GradCAM (RRR-G) (Schramowskiet al., 2020)
* Human Importance-aware Network Tuning (HINT) (Selvaraju et. al., 2019)
* Contextual Decomposition Explanation Penalization (CDEP) (Rieger et. al., 2019)
* Counterexamples (CE) (Teso & Kersting, 2019).


## How to use it?
Install the `requirements.txt`. Then train a model for a certain XIL method and dataset, e.g. use `main_MNIST.py`. Next, evaluate the trained model with `wrong_reason_MNIST.py` using our proposed WR metric. `switch_xil_on.py`, `interaction_efficiency.py`, and `robustness.ipnyb` can be used to further evaluate a XIL-revised model. With `visualize_explanations.ipynb` heatmaps can be generated and visualized in order to get qualitative results.

## Framework structure
The following describes and explains the core components/modules:

* `xil_methods`: Implements the XIL loss functions in `xil_loss.py` and functions for the generation of counterexamples in `ce.py`. 
* `learner`: Package that contains classes and functions related to a ML learner. `models/dnns.py` implements different neural networks (SimpleConvNET, VGG16). `learner.py` implements a ML learner with different properties and provides the training routines (`fit()`, `fit_isic()`, `fit_n_expl_shuffled_dataloader()`) as well as some utility functions for scoring/storing/loading a learner. Trained models are stored in the `model_store`.
* `data_store`: Place for all dataset-related stuff. `datasets.py` currently implements three datasets (DecoyMNIST, DecoyFMNIST and ISIC Skin Cancer 2019). `rawdata` contains utility functions for setting up and downloading raw data. 
* `explainer.py`: Collection of explainer methods used to visualize explanations. Uses the captum library for the GradCAM method. Provides functions to quantify Wrong Reason (WR metric) for IG, LIME and GradCAM. 
* `util.py`: Utilzation functions.
* `main_MNIST.py`: Implements the main setup for the DecoyMNIST/FMNIST experiments..  
* `main_isic.py`: Implements the main setup for the ISIC experiments.
* `wrong_reason.py`: Example file to quantify WR and plot heatmaps.
* `runs`: Folders to store outputs 

## Contact
**Author:** Felix Friedrich  
**Institution:** Technische Universität Darmstadt (2022)  
**Department:** Machine Learning Lab, Computer Science, TU Darmstadt, Germany   
**Mail:** <friedrich@cs.tu-darmstadt.de>

## Citation
If you like or use our work please cite us:
```bibtex
@article{friedrich2022XIL_typo,
      title={A Typology to Explore and Guide Explanatory Interactive Machine Learning}, 
      author={Felix Friedrich and Wolfgang Stammer and Patrick Schramowski and Kristian Kersting},
      year={2022},
      journal={arXiv preprint arXiv:2203.03668}
}
```
<!-- #endregion -->

```python

```
