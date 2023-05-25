# -*- coding: utf-8 -*-
"""Collection of different XIL loss classes based on PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from captum.attr import LayerGradCam, LayerAttribution
import numpy as np

from xil_methods.cd_util import cd
import util
import explainer


"""
All module's forward() functions have the same parameters, making it easier to iterate through collections of instances.

This comes at the cost of providing some functions with unneccessary arguments.
"""


class RRRLoss(nn.Module):
    """
    Right for the Right Reason loss (RRR) as proposed by Ross et. al (2017) with minor changes.
    See https://arxiv.org/abs/1703.03717. 
    The RRR loss calculates the Input Gradients as prediction explanation and compares it
    with the (ground-truth) user explanation.

    """

    def __init__(self, normalization_rate, regularization_rate, base_criterion=F.cross_entropy, rr_clipping=None):
        """
        Args:
            regularizer_rate: controls the influence of the right reason loss.
            base_criterion: criterion to use for right answer l
            weight: if specified then weight right reason loss by classes. Tensor
                with shape (c,) c=classes. WARNING !! Currently only working for 
                the special case that whole X in fwd has the same class (as is the
                case in isic 2019).
            rr_clipping: clip the RR loss to a maximum per batch.   
        """
        super().__init__()
        self.normalization_rate = normalization_rate
        self.regularization_rate = regularization_rate
        self.base_criterion = base_criterion
        # self.weight = weight
        self.rr_clipping = rr_clipping

    def forward(self, model, X, y, ra_loss, E_pnlt, E_rwrd, logits, device=None, mask=None,):
        """
        Returns Right-Reason loss

        Args:
            model: pytorch model.
            X: inputs (train set).
            y: Ground-truth labels.
            ra_loss: Right-Answer loss predicting y through X.
            E_pnlt: Explanation annotations matrix (ones penalize regions).
            E_rwrd: Explanation annotations matrix (ones reward regions).
            logits: model output logits with input X.
            device: either 'cpu' or 'cuda' 
        """
        # get gradients w.r.t. to the input
        log_prob_ys = F.log_softmax(logits, dim=1)
        log_prob_ys.retain_grad()
        gradXes = torch.autograd.grad(log_prob_ys, X, torch.ones_like(log_prob_ys),
                                      create_graph=True, allow_unused=True)[0]

        # if expl.shape [n,1,h,w] and gradXes.shape [n,3,h,w] then torch broadcasting
        # is used implicitly
        A_gradX = torch.mul(E_pnlt, gradXes) ** 2

        if mask is not None:
            for i in range(len(A_gradX)):
                A_gradX[i] = mask[i] * A_gradX[i]
            # print("!!! MASK NOT NONE !!!")

        right_reason_loss = torch.sum(A_gradX)
        right_reason_loss *= self.normalization_rate
        right_reason_loss *= self.regularization_rate

        # if self.weight is not None:
        #     right_reason_loss *= [y[0]]

        if self.rr_clipping:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = self.rr_clipping

        return right_reason_loss


class RBRLoss(nn.Module):
    """
    Right for the Better Reasons (RBR) loss according to Shao et. al 2021.
    Using identity matrix instead of hessian.
    """

    def __init__(self, normalization_rate, regularization_rate, base_criterion=F.cross_entropy,
                 rr_clipping=None):
        """
        Args:
            regularizer_rate: controls the influence of the right reason loss.
            base_criterion: criterion to use for right answer loss
            weight: if specified then weight right reason loss by classes. Tensor
                with shape (c,) c=classes. WARNING !! Currently only working for 
                the special case that whole X in fwd has the same class (as is the
                case in isic 2019).
            rr_clipping: sets the max right reason loss to specified value -> Helps smoothing 
                and stabilizing training process.    
        """
        super().__init__()
        self.normalization_rate = normalization_rate
        self.regularization_rate = regularization_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping  # good rate for decoy mnist 1.0
        # self.weight = weight

    def forward(self, model, X, y, ra_loss, E_pnlt, E_rwrd, logits, device=None, mask=None):
        """
        Returns Right-Reason loss

        Args:
            model: pytorch model.
            X: inputs (train set).
            y: Ground-truth labels.
            ra_loss: Right-Answer loss predicting y through X.
            E_pnlt: Explanation annotations matrix (ones penalize regions).
            E_rwrd: Explanation annotations matrix (ones reward regions).
            logits: model output logits with input X.
            device: either 'cpu' or 'cuda' 
        """
        # use information from the influence function (IF)
        # to compute saliency maps of features and penalize features according to expl masks

        # GET gradients of Influnece function
        # get loss gradients wrt to model params
        ra_loss.retain_grad()
        loss_grads_wrt_model_params_all = torch.autograd.grad(ra_loss, model.parameters(),
                                                              torch.ones_like(ra_loss), create_graph=True, allow_unused=True)
        # currently the grads are a list of every grad loss for all params wrt to the layer
        # --> we need to get the sum of all grads
        loss_grads_wrt_model_params = torch.sum(
            (torch.cat([t.flatten() for t in loss_grads_wrt_model_params_all])))
        loss_grads_wrt_model_params.retain_grad()

        # get loss gradients wrt to input x
        if_grads = torch.autograd.grad(loss_grads_wrt_model_params, X,
                                       torch.ones_like(loss_grads_wrt_model_params), create_graph=True, allow_unused=True)[0]

        # get grads of Input Gradients
        # get gradients w.r.t. to the input
        log_prob_ys = F.log_softmax(logits, dim=1)
        log_prob_ys.retain_grad()
        ig_grads = torch.autograd.grad(log_prob_ys, X, torch.ones_like(log_prob_ys),
                                       create_graph=True, allow_unused=True)[0]

        # Right reason = regularizer x (IF grads x IG grads)**2

        grads = torch.mul(if_grads, ig_grads)
        A_gradX = torch.mul(E_pnlt, grads) ** 2

        if mask is not None:
            for i in range(len(A_gradX)):
                A_gradX[i] = mask[i] * A_gradX[i]
            # print("!!! MASK NOT NONE !!!")

        right_reason_loss = torch.sum(A_gradX)
        right_reason_loss *= self.normalization_rate
        right_reason_loss *= self.regularization_rate

        # if self.weight is not None:
        #     right_reason_loss *= self.weight[y[0]]

        if self.rr_clipping:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = self.rr_clipping

        return right_reason_loss


class RRRGradCamLoss(nn.Module):
    """
    RRRGradCAM loss. Similar to the RRR loss but instead of IG uses
    GradCAM as explainer method for the prediction.

    Note: Can only be applied to CNNs.
    """

    def __init__(self, normalization_rate, regularization_rate, base_criterion=F.cross_entropy, reduction='sum',
                 last_conv_specified=False, rr_clipping=None):
        """
        Args:
            regularizer_rate: controls the influence of the right reason loss.
            base_criterion: criterion to use for right answer loss
            reduction: Method to reduce loss. Either 'sum' or 'mean'.
            last_conv_specified: if True then uses the last convolutional layer
                which must have the name 'last_conv' in the network definition. If
                False then the last conv layer is calculated dynamically every time
                (increases run time).
            weight: if specified then weight right reason loss by classes. Tensor
                with shape (c,) c=classes.
            rr_clipping: sets the max right reason loss to specified value -> Helps smoothing 
                and stabilizing training process.    
        """
        super().__init__()
        self.normalization_rate = normalization_rate
        self.regularization_rate = regularization_rate
        self.base_criterion = base_criterion
        self.reduction = reduction
        self.last_conv_specified = last_conv_specified
        # self.weight = weight
        self.rr_clipping = rr_clipping

    def forward(self, model, X, y, ra_loss, E_pnlt, E_rwrd, logits, device, mask=None):
        """
        Returns Right-Reason loss

        Args:
            model: pytorch model.
            X: inputs (train set).
            y: Ground-truth labels.
            ra_loss: Right-Answer loss predicting y through X.
            E_pnlt: Explanation annotations matrix (ones penalize regions).
            E_rwrd: Explanation annotations matrix (ones reward regions).
            logits: model output logits with input X.
            device: either 'cpu' or 'cuda' 
        """
        # get gradients w.r.t. to the input
        log_ys = torch.argmax(F.softmax(logits, dim=1), dim=1)

        model.eval()
        # network importance score --> compute GradCam attribution of last conv layer
        if self.last_conv_specified:
            explainer = LayerGradCam(model, model.last_conv)
        else:
            last_conv_layer = util.get_last_conv_layer(model)
            explainer = LayerGradCam(model, last_conv_layer)

        saliencies = explainer.attribute(
            X, target=log_ys, relu_attributions=False)
        # apply relu by hand, or check neg values for GradCam meaning
        # normalize grads [0-1] to compare them to expl masks --> includes pos and neg values
        norm_saliencies = util.norm_saliencies_fast(saliencies)

        # downsample expl masks to match saliency masks
        h, w = norm_saliencies.shape[2], norm_saliencies.shape[3]
        downsampled_expl = transforms.Resize((h, w))(E_pnlt)

        attr = torch.mul(downsampled_expl, norm_saliencies) ** 2

        right_reason_loss = torch.zeros(1,).to(device)

        # if self.weight is not None:
        #     attr = torch.sum(attr, dim=(1, 2, 3))
        #     for i in range(len(self.weight)):
        #         class_indices_i = torch.nonzero((y == i), as_tuple=True)[0]
        #         attr[class_indices_i] *= self.weight[i]

        if mask is not None:
            for i in range(len(attr)):
                attr[i] = mask[i] * attr[i]
            # print("!!! MASK NOT NONE !!!")

        if self.reduction == 'sum':
            right_reason_loss = torch.sum(attr)
        elif self.reduction == 'mean':
            right_reason_loss = torch.sum(attr) / len(X)

        right_reason_loss *= self.normalization_rate
        right_reason_loss *= self.regularization_rate

        if self.rr_clipping:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = self.rr_clipping

        return right_reason_loss


class CDEPLoss(nn.Module):
    """
    CDEP loss as proposed by Rieger et. al 2020.
    See https://github.com/laura-rieger/deep-explanation-penalization.
    """

    def __init__(self, normalization_rate, regularization_rate, base_criterion=F.cross_entropy,
                 model_type='mnist', rr_clipping=None):
        """
        Args:
            regularizer_rate: controls the influence of the right reason loss.
            base_criterion: criterion to use for right answer loss
            weight: if specified then weight right reason loss by classes. Tensor
                with shape (c,) c=classes. WARNING !! Currently only working for 
                the special case that whole X in fwd has the same class (as is the
                case in isic 2019).
            rr_clipping: sets the max right reason loss to specified value -> Helps smoothing 
                and stabilizing training process.
            model_type: specify the network architecture. Either 'mnist' or 'vgg'   
        """
        super().__init__()
        self.normalization_rate = normalization_rate
        self.regularization_rate = regularization_rate
        self.base_criterion = base_criterion
        # self.weight = weight
        self.model_type = model_type
        self.rr_clipping = rr_clipping

    def forward(self, model, X, y, ra_loss, E_pnlt, E_rwrd, logits, device):
        """
        Returns Right-Reason loss

        Args:
            model: pytorch model.
            X: inputs (train set).
            y: Ground-truth labels.
            ra_loss: Right-Answer loss predicting y through X.
            E_pnlt: Explanation annotations matrix (ones penalize regions).
            E_rwrd: Explanation annotations matrix (ones reward regions).
            logits: model output logits with input X.
            device: either 'cpu' or 'cuda' 
        """
        # rel, irrel = cd.cd(expl, X, model, model_type=model_type, device=device)
        right_reason_loss = torch.zeros(1,).to(device)

        # calculate Contextual Decompostions (CD)
        rel, irrel = cd.cd(X, model, E_pnlt, device=device,
                           model_type=self.model_type)

        right_reason_loss += F.softmax(torch.stack(
            (rel.view(-1), irrel.view(-1)), dim=1), dim=1)[:, 0].mean()

        right_reason_loss *= self.normalization_rate
        right_reason_loss *= self.regularization_rate

        # if self.weight is not None:
        #     right_reason_loss *= self.weight[y[0]]

        if self.rr_clipping:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = self.rr_clipping

        return torch.squeeze(right_reason_loss)


class HINTLoss(nn.Module):
    """
    Simplified version of HINT loss based on Selvaraju et. al 2020.
    See https://arxiv.org/abs/1902.03751. 

    Conceptually, this is the equivalent of the RRRGradCAM but instead
    of penalizing wrong reason it rewards right reason.
    """

    def __init__(self, normalization_rate, regularization_rate, base_criterion=F.cross_entropy, reduction='sum',
                 last_conv_specified=False, upsample=False, positive_only=False, rr_clipping=None):
        """
        Args:
            regularizer_rate: controls the influence of the right reason loss.
            base_criterion: criterion to use for right answer loss
            reduction: reduction method either 'none', 'mean', 'sum'.
            last_conv_specified: if True then uses the last convolutional layer
                which must have the name 'last_conv' in the network definition. If
                False then the last conv layer is calculated dynamically every time
                (increases run time).
            upsample: if True then the saliency masks of the model are upsampled to match
                the user explanation masks. If False then the user expl masks are downsampled.
            weight: if specified then weight right reason loss by classes. Tensor
                with shape (c,) c=classes.
            positive_only: if True all negative attribution gets zero.
            rr_clipping: sets the max right reason loss to specified value -> Helps smoothing 
                and stabilizing training process. 
        """
        super().__init__()
        self.normalization_rate = normalization_rate
        self.regularization_rate = regularization_rate
        self.base_criterion = base_criterion
        self.reduction = reduction
        self.last_conv_specified = last_conv_specified
        self.upsample = upsample
        # self.weight = weight
        self.positive_only = positive_only
        self.rr_clipping = rr_clipping

    def forward(self, model, X, y, ra_loss, E_pnlt, E_rwrd, logits, device, mask=None):
        """
        Returns Right-Reason loss

        Args:
            model: pytorch model.
            X: inputs (train set).
            y: Ground-truth labels.
            ra_loss: Right-Answer loss predicting y through X.
            E_pnlt: Explanation annotations matrix (ones penalize regions).
            E_rwrd: Explanation annotations matrix (ones reward regions).
            logits: model output logits with input X.
            device: either 'cpu' or 'cuda' 
        """
        # human importance map = E_rwrd: -> array {0,1} region with high importance have ones
        model.eval()

        # network importance score --> compute GradCam attribution of last conv layer
        if self.last_conv_specified:
            explainer = LayerGradCam(model, model.last_conv)
        else:
            last_conv_layer = util.get_last_conv_layer(model)
            explainer = LayerGradCam(model, last_conv_layer)

        saliencies = explainer.attribute(X, target=y, relu_attributions=False)
        # normalize grads [0-1] to compare them to expl masks
        # norm_saliencies = util.norm_saliencies(saliencies)
        norm_saliencies = util.norm_saliencies_fast(
            saliencies, positive_only=self.positive_only)

        right_reason_loss = torch.zeros(1,).to(device)

        if self.upsample:  # upsample saliency masks to match expl masks
            # resize grad attribution to match explanation size
            h, w = E_rwrd.shape[2], E_rwrd.shape[3]
            upsampled_saliencies = LayerAttribution.interpolate(
                norm_saliencies, (h, w))
            if mask is not None:
                for i in range(len(E_rwrd)):
                    upsampled_saliencies[i] = mask[i] * upsampled_saliencies[i]
                    E_rwrd[i] = mask[i] * E_rwrd[i]
                # print("!!! MASK NOT NONE !!!")
            attr = F.mse_loss(upsampled_saliencies, E_rwrd,
                              reduction=self.reduction)

        else:  # downsample expl masks to match saliency masks
            h, w = norm_saliencies.shape[2], norm_saliencies.shape[3]
            downsampled_expl = transforms.Resize((h, w))(E_rwrd)
            if mask is not None:
                for i in range(len(E_rwrd)):
                    downsampled_expl[i] = mask[i] * downsampled_expl[i]
                    norm_saliencies[i] = mask[i] * norm_saliencies[i]
                # print("!!! MASK NOT NONE !!!")
            # right_reason_loss += F.mse_loss(norm_saliencies, downsampled_expl, reduction='none')
            attr = F.mse_loss(norm_saliencies, downsampled_expl,
                              reduction=self.reduction)

        # if self.weight is not None and self.reduction == 'none':
        #     attr = torch.sum(attr, dim=(1, 2, 3))
        #     for i in range(len(self.weight)):
        #         class_indices_i = torch.nonzero((y == i), as_tuple=True)[0]
        #         attr[class_indices_i] *= self.weight[i]

        if self.reduction == 'sum':
            right_reason_loss += torch.sum(attr)
        elif self.reduction == 'mean':
            right_reason_loss += torch.sum(attr) / len(X)

        right_reason_loss *= self.normalization_rate
        right_reason_loss *= self.regularization_rate

        # Human-Network Importance Alignment via loss
        if self.rr_clipping:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = self.rr_clipping

        model.train()  # probably useless
        return torch.squeeze(right_reason_loss)


class HINTLoss_IG(nn.Module):
    """
    HINT Loss extended version with IG instead of GradCam
    """

    def __init__(self, normalization_rate, regularization_rate, base_criterion=F.cross_entropy, reduction='sum', rr_clipping=None):
        """
        Args:
            regularizer_rate: controls the influence of the right reason loss.
            base_criterion: criterion to use for right answer loss
            reduction: reduction method either 'none', 'mean', 'sum'.
            last_conv_specified: if True then uses the last convolutional layer
                which must have the name 'last_conv' in the network definition. If
                False then the last conv layer is calculated dynamically every time
                (increases run time).
            upsample: if True then the saliency masks of the model are upsampled to match
                the user explanation masks. If False then the user expl masks are downsampled.
            positive_only: if True all negative attribution gets zero.
            rr_clipping: sets the max right reason loss to specified value -> Helps smoothing
                and stabilizing training process.
        """
        super().__init__()
        self.normalization_rate = normalization_rate
        self.regularization_rate = regularization_rate
        self.base_criterion = base_criterion
        self.reduction = reduction
        self.rr_clipping = rr_clipping

    def forward(self, model, X, y, ra_loss, E_pnlt, E_rwrd, logits, device):
        """
        Returns Right-Reason loss

        Args:
            model: pytorch model.
            X: inputs (train set).
            y: Ground-truth labels.
            ra_loss: Right-Answer loss predicting y through X.
            E_pnlt: Explanation annotations matrix (ones penalize regions).
            E_rwrd: Explanation annotations matrix (ones reward regions).
            logits: model output logits with input X.
            device: either 'cpu' or 'cuda' 
        """
        model.eval()

        # get gradients w.r.t. to the input
        log_prob_ys = F.log_softmax(logits, dim=1)
        log_prob_ys.retain_grad()
        gradXes = torch.autograd.grad(log_prob_ys, X, torch.ones_like(
            log_prob_ys), create_graph=True, allow_unused=True)[0]

        A_gradX = F.mse_loss(gradXes, E_rwrd, reduction=self.reduction)

        right_reason_loss = torch.zeros(1,).to(device)

        if self.reduction == 'sum':
            right_reason_loss += torch.sum(A_gradX)
        elif self.reduction == 'mean':
            right_reason_loss += torch.sum(A_gradX) / len(X)

        right_reason_loss *= self.normalization_rate
        right_reason_loss *= self.regularization_rate

        # Human-Network Importance Alignment via loss
        if self.rr_clipping:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = self.rr_clipping

        model.train()  # probably useless
        return torch.squeeze(right_reason_loss)
