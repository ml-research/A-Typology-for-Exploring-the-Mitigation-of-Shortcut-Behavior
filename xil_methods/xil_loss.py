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


class RRRLoss(nn.Module):
    """
    Right for the Right Reason loss (RRR) as proposed by Ross et. al (2017) with minor changes.
    See https://arxiv.org/abs/1703.03717. 
    The RRR loss calculates the Input Gradients as prediction explanation and compares it
    with the (ground-truth) user explanation.
    
    """
    def __init__(self, regularizer_rate=10, base_criterion=F.cross_entropy, weight=None,\
         rr_clipping=None):
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
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.weight = weight
        self.rr_clipping = rr_clipping 

    def forward(self, X, y, expl, logits, mask=None):
        """
        Returns (loss, right_answer_loss, right_reason_loss)

        Args:
            X: inputs.
            y: ground-truth labels.
            expl: explanation annotations masks (ones penalize regions).
            logits: model output logits. 
        """
        # get gradients w.r.t. to the input
        log_prob_ys = F.log_softmax(logits, dim=1)
        log_prob_ys.retain_grad()
        gradXes = torch.autograd.grad(log_prob_ys, X, torch.ones_like(log_prob_ys), \
            create_graph=True, allow_unused=True)[0]
        
        # if expl.shape [n,1,h,w] and gradXes.shape [n,3,h,w] then torch broadcasting 
        # is used implicitly
        A_gradX = torch.mul(expl, gradXes) ** 2

        if mask is not None:
            for i in range(len(A_gradX)):
                A_gradX[i] = mask[i] * A_gradX[i]
            # print("!!! MASK NOT NONE !!!")

        right_reason_loss = torch.sum(A_gradX)
        right_reason_loss *= self.regularizer_rate

        if self.weight is not None:
            right_reason_loss *= self.weight#[y[0]]

        if self.rr_clipping is not None:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = right_reason_loss - right_reason_loss + self.rr_clipping

        return right_reason_loss

class RBRLoss(nn.Module):
    """
    Right for the Better Reasons (RBR) loss according to Shao et. al 2021.
    Using identity matrix instead of hessian.
    """
    def __init__(self, regularizer_rate=100000, base_criterion=F.cross_entropy, \
            rr_clipping=None, weight=None):
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
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping # good rate for decoy mnist 1.0
        self.weight = weight

    def forward(self, model, X, y, expl, logits, mask=None):
        """
        Returns (loss, right_answer_loss, right_reason_loss)

        Args:
            model: pytorch model.
            X: inputs.
            y: ground-truth labels.
            expl: explanation annotations masks (ones penalize regions).
            logits: model output logits. 
        """
        # Calculate right answer loss
        right_answer_loss = self.base_criterion(logits, y)

        # use information from the influence function (IF) 
        # to compute saliency maps of features and penalize features according to expl masks

        ################## GET gradients of Influnece function
        # get loss gradients wrt to model params
        right_answer_loss.retain_grad()
        loss_grads_wrt_model_params_all = torch.autograd.grad(right_answer_loss, model.parameters(), \
            torch.ones_like(right_answer_loss), create_graph=True, allow_unused=True)
        # currently the grads are a list of every grad loss for all params wrt to the layer 
        # --> we need to get the sum of all grads
        loss_grads_wrt_model_params = torch.sum((torch.cat([t.flatten() for t in loss_grads_wrt_model_params_all])))
        loss_grads_wrt_model_params.retain_grad()

        # get loss gradients wrt to input x 
        if_grads = torch.autograd.grad(loss_grads_wrt_model_params, X, \
            torch.ones_like(loss_grads_wrt_model_params), create_graph=True, allow_unused=True)[0]

        ######### get grads of Input Gradients
        # get gradients w.r.t. to the input
        log_prob_ys = F.log_softmax(logits, dim=1)
        log_prob_ys.retain_grad()
        ig_grads = torch.autograd.grad(log_prob_ys, X, torch.ones_like(log_prob_ys), \
            create_graph=True, allow_unused=True)[0]

        # Right reason = regularizer x (IF grads x IG grads)**2

        grads = torch.mul(if_grads, ig_grads)
        A_gradX = torch.mul(expl, grads) ** 2

        if mask is not None:
            for i in range(len(A_gradX)):
                A_gradX[i] = mask[i] * A_gradX[i]
            # print("!!! MASK NOT NONE !!!")

        right_reason_loss = torch.sum(A_gradX)
        right_reason_loss *= self.regularizer_rate

        if self.weight is not None:
            right_reason_loss *= self.weight[y[0]]

        if self.rr_clipping is not None:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = right_reason_loss - right_reason_loss + self.rr_clipping

        return right_reason_loss

class RRRGradCamLoss(nn.Module):
    """
    RRRGradCAM loss. Similar to the RRR loss but instead of IG uses
    GradCAM as explainer method for the prediction.

    Note: Can only be applied to CNNs.
    """

    def __init__(self, regularizer_rate=1, base_criterion=F.cross_entropy, reduction='sum',\
        last_conv_specified=False, weight=None, rr_clipping=None):
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
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.reduction = reduction
        self.last_conv_specified = last_conv_specified
        self.weight = weight
        self.rr_clipping = rr_clipping

    def forward(self, model, X, y, expl, logits, device, mask=None):
        """
        Returns right_answer_loss

        Args:
            X: inputs.
            y: ground-truth labels.
            expl: explanation annotations matrix (ones penalize regions).
            logits: model output logits. 
            device:
            mask:
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

        saliencies = explainer.attribute(X, target=log_ys, relu_attributions=False)
        # apply relu by hand, or check neg values for GradCam meaning
        # normalize grads [0-1] to compare them to expl masks --> includes pos and neg values
        norm_saliencies = util.norm_saliencies_fast(saliencies)
        
        # downsample expl masks to match saliency masks
        h, w = norm_saliencies.shape[2], norm_saliencies.shape[3]
        downsampled_expl = transforms.Resize((h,w))(expl)
        
        attr = torch.mul(downsampled_expl, norm_saliencies) ** 2

        right_reason_loss = torch.zeros(1,).to(device)

        if self.weight is not None:
            attr = torch.sum(attr, dim=(1,2,3))
            for i in range(len(self.weight)):
                class_indices_i = torch.nonzero((y == i), as_tuple=True)[0]
                attr[class_indices_i] *= self.weight[i]

        if mask is not None:
            for i in range(len(attr)):
                attr[i] = mask[i] * attr[i]
            # print("!!! MASK NOT NONE !!!")

        if self.reduction == 'sum':
            right_reason_loss = torch.sum(attr)
        elif self.reduction == 'mean':
            right_reason_loss = torch.sum(attr) / len(X)

        right_reason_loss *= self.regularizer_rate


        if self.rr_clipping is not None:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = right_reason_loss - right_reason_loss + self.rr_clipping

        return right_reason_loss

class CDEPLoss(nn.Module):
    """
    CDEP loss as proposed by Rieger et. al 2020.
    See https://github.com/laura-rieger/deep-explanation-penalization.
    """
    def __init__(self, regularizer_rate=1000000, base_criterion=F.cross_entropy, weight=None, \
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
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.weight = weight
        self.model_type = model_type
        self.rr_clipping = rr_clipping

    def forward(self, model, X, y, expl, device):
        """
        Returns right_answer_loss

        Args:
            model: pytorch model.
            X: inputs (train set).
            y: Ground-truth labels.
            expl: Explanation annotations matrix (ones penalize regions).
        """
        #rel, irrel = cd.cd(expl, X, model, model_type=model_type, device=device)
        right_reason_loss = torch.zeros(1,).to(device)
        
        # calculate Contextual Decompostions (CD)
        rel, irrel = cd.cd(X, model, expl, device=device, model_type=self.model_type)

        right_reason_loss += F.softmax(torch.stack((rel.view(-1), irrel.view(-1)), dim=1), dim=1)[:, 0].mean()

        right_reason_loss *= self.regularizer_rate

        if self.weight is not None:
            right_reason_loss *= self.weight[y[0]]

        if self.rr_clipping is not None:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = right_reason_loss  - right_reason_loss + self.rr_clipping

        return torch.squeeze(right_reason_loss)

class HINTLoss(nn.Module):
    """
    Simplified version of HINT loss based on Selvaraju et. al 2020.
    See https://arxiv.org/abs/1902.03751. 

    Conceptually, this is the equivalent of the RRRGradCAM but instead
    of penalizing wrong reason it rewards right reason.
    """

    def __init__(self, regularizer_rate=100, base_criterion=F.cross_entropy, reduction='sum', \
        last_conv_specified=False, upsample=False, weight=None, positive_only=False, rr_clipping=None):
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
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.reduction = reduction
        self.last_conv_specified = last_conv_specified
        self.upsample = upsample
        self.weight = weight
        self.positive_only = positive_only
        self.rr_clipping = rr_clipping


    def forward(self, model, X, y, expl, device, mask=None):
        """
        Returns (loss, right_answer_loss, right_reason_loss)

        Args:
            model: pytorch model.
            X: inputs (train set).
            y: Ground-truth labels.
            expl: Explanation annotations matrix (ones penalize regions).
            logits: model output logits.
            device: either 'cpu' or 'cuda' 
        """
        # human importance map = expl: -> array {0,1} region with high importance have ones
        # calculate right answer loss 
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
        norm_saliencies = util.norm_saliencies_fast(saliencies, positive_only=self.positive_only)

        right_reason_loss = torch.zeros(1,).to(device)

        if self.upsample: # upsample saliency masks to match expl masks
            # resize grad attribution to match explanation size 
            h, w = expl.shape[2], expl.shape[3]
            upsampled_saliencies = LayerAttribution.interpolate(norm_saliencies, (h, w))
            if mask is not None:
                for i in range(len(expl)):
                    upsampled_saliencies[i] = mask[i] * upsampled_saliencies[i]
                    expl[i] = mask[i] * expl[i]
                # print("!!! MASK NOT NONE !!!")
            attr = F.mse_loss(upsampled_saliencies, expl, reduction=self.reduction)
        
        else: # downsample expl masks to match saliency masks
            h, w = norm_saliencies.shape[2], norm_saliencies.shape[3]
            downsampled_expl = transforms.Resize((h,w))(expl)
            if mask is not None:
                for i in range(len(expl)):
                    downsampled_expl[i] = mask[i] * downsampled_expl[i]
                    norm_saliencies[i] = mask[i] * norm_saliencies[i]
                # print("!!! MASK NOT NONE !!!")
            #right_reason_loss += F.mse_loss(norm_saliencies, downsampled_expl, reduction='none')
            attr = F.mse_loss(norm_saliencies, downsampled_expl, reduction=self.reduction)

        if self.weight is not None and self.reduction == 'none':
            attr = torch.sum(attr, dim=(1,2,3))
            for i in range(len(self.weight)):
                class_indices_i = torch.nonzero((y == i), as_tuple=True)[0]
                attr[class_indices_i] *= self.weight[i]
        
        if self.reduction == 'sum': 
            right_reason_loss += torch.sum(attr)
        elif self.reduction == 'mean':
            right_reason_loss += torch.sum(attr) / len(X)

        right_reason_loss *= self.regularizer_rate

        # Human-Network Importance Alignment via loss
        if self.rr_clipping is not None:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = right_reason_loss  - right_reason_loss + self.rr_clipping

        model.train() # probably useless
        return torch.squeeze(right_reason_loss)

class HINTLoss_IG(nn.Module):
    """
    HINT Loss extended version with IG instead of GradCam
    """

    def __init__(self, regularizer_rate=50000, base_criterion=F.cross_entropy, reduction='sum', rr_clipping=None):
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
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.reduction = reduction
        self.rr_clipping = rr_clipping

    def forward(self, model, X, y, expl, logits, device):
        # calculate right answer loss
        right_answer_loss = self.base_criterion(logits, y)
        model.eval()

        # get gradients w.r.t. to the input
        log_prob_ys = F.log_softmax(logits, dim=1)
        log_prob_ys.retain_grad()
        gradXes = torch.autograd.grad(log_prob_ys, X, torch.ones_like(log_prob_ys), create_graph=True, allow_unused=True)[0]

        A_gradX = F.mse_loss(gradXes, expl, reduction=self.reduction)

        right_reason_loss = torch.zeros(1,).to(device)

        if self.reduction == 'sum':
            right_reason_loss += torch.sum(A_gradX)
        elif self.reduction == 'mean':
            right_reason_loss += torch.sum(A_gradX) / len(X)

        right_reason_loss *= self.regularizer_rate

        # Human-Network Importance Alignment via loss
        if self.rr_clipping is not None:
            if right_reason_loss > self.rr_clipping:
                right_reason_loss = right_reason_loss - right_reason_loss + self.rr_clipping

        res = right_reason_loss + right_answer_loss
        model.train()  # probably useless
        return res, right_answer_loss, right_reason_loss

class MixLoss1(nn.Module):
    """
    MixLoss1 = RRR + RBR + RRRG
    """
    def __init__(self, regularizer_rate = None, base_criterion=F.cross_entropy, \
            rr_clipping=None, weight=None, last_conv_specified=False, reduction='sum', regrate_rrr=10, regrate_rbr=100000, regrate_rrrg=1):

        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping # good rate for decoy mnist 1.0
        self.weight = weight
        self.last_conv_specified = last_conv_specified
        self.reduction = reduction
        self.regrate_rrr = regrate_rrr
        self.regrate_rbr = regrate_rbr
        self.regrate_rrrg = regrate_rrrg


    def forward(self, model, X, y, expl, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_rrr
        RRR = RRRLoss.forward(self, X, y, expl, logits)
        right_answer_loss += RRR[1]
        right_reason_loss += RRR[2]

        self.regularizer_rate = self.regrate_rbr
        RBR = RBRLoss.forward(self, model, X, y, expl.float(), logits)
        right_answer_loss += RBR[1]
        right_reason_loss += RBR[2]

        self.regularizer_rate = self.regrate_rrrg
        RRRG = RRRGradCamLoss.forward(self, model, X, y, expl.float(), logits, device)
        right_answer_loss += RRRG[1]
        right_reason_loss += RRRG[2]

        right_answer_loss /= 3

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss2(nn.Module):
    """
        MixLoss2 = RRRG + HINT
    """
    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, reduction='sum',\
        last_conv_specified=False, weight=None, rr_clipping=None, upsample=False, positive_only=False, regrate_rrrg = 1, regrate_hint = 100):

        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping # good rate for decoy mnist 1.0
        self.weight = weight
        self.last_conv_specified = last_conv_specified
        self.reduction = reduction
        self.upsample = upsample
        self.positive_only = positive_only
        self.regrate_rrrg = regrate_rrrg
        self.regrate_hint = regrate_hint

    def forward(self, model, X, y, expl_p, expl_r, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_rrrg
        RRRG = RRRGradCamLoss.forward(self, model, X, y, expl_p, logits, device)
        right_answer_loss += RRRG[1]
        right_reason_loss += RRRG[2]

        self.regularizer_rate = self.regrate_hint
        self.last_conv_specified = True
        self.upsample = True
        HINT = HINTLoss.forward(self, model, X, y, expl_r, logits, device)
        right_answer_loss += HINT[1]
        right_reason_loss += HINT[2][0]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss3(nn.Module):
    """
        MixLoss3 = RRR + CDEP
    """
    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, weight=None, \
                 model_type='mnist', rr_clipping=None, regrate_rrr = 10, regrate_cdep = 1000000):
        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping  # good rate for decoy mnist 1.0
        self.weight = weight
        self.model_type = model_type
        self.regrate_rrr = regrate_rrr
        self.regrate_cdep = regrate_cdep

    def forward(self, model, X, y, expl, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_rrr
        RRR = RRRLoss.forward(self, X, y, expl, logits)
        right_answer_loss += RRR[1]
        right_reason_loss += RRR[2]

        self.regularizer_rate = self.regrate_cdep
        CDEP = CDEPLoss.forward(self, model, X, y, expl.float(), logits, device)
        right_answer_loss += CDEP[1]
        right_reason_loss += CDEP[2][0]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss4(nn.Module):
    """
        MixLoss4 = RRR + RBR
    """

    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, \
                 rr_clipping=None, weight=None, regrate_rrr=10,
                 regrate_rbr=100000):
        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping  # good rate for decoy mnist 1.0
        self.weight = weight
        self.regrate_rrr = regrate_rrr
        self.regrate_rbr = regrate_rbr

    def forward(self, model, X, y, expl, logits):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_rrr
        RRR = RRRLoss.forward(self, X, y, expl, logits)
        right_answer_loss += RRR[1]
        right_reason_loss += RRR[2]

        self.regularizer_rate = self.regrate_rbr
        RBR = RBRLoss.forward(self, model, X, y, expl.float(), logits)
        right_answer_loss += RBR[1]
        right_reason_loss += RBR[2]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss5(nn.Module):
    """
        MixLoss5 = RBR + CDEP
    """
    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, weight=None, \
                 model_type='mnist', rr_clipping=None, regrate_rbr = 100000, regrate_cdep = 1000000):
        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping  # good rate for decoy mnist 1.0
        self.weight = weight
        self.model_type = model_type
        self.regrate_rbr = regrate_rbr
        self.regrate_cdep = regrate_cdep

    def forward(self, model, X, y, expl, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_rbr
        RBR = RBRLoss.forward(self, model, X, y, expl.float(), logits)
        right_answer_loss += RBR[1]
        right_reason_loss += RBR[2]

        self.regularizer_rate = self.regrate_cdep
        CDEP = CDEPLoss.forward(self, model, X, y, expl.float(), logits, device)
        right_answer_loss += CDEP[1]
        right_reason_loss += CDEP[2][0]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss6(nn.Module):
    """
        MixLoss6 = RRRG + CDEP
    """
    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, reduction='sum', \
                 last_conv_specified=False, weight=None, rr_clipping=None, model_type='mnist', regrate_rrrg = 1, regrate_cdep = 1000000):
        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.reduction = reduction
        self.last_conv_specified = last_conv_specified
        self.weight = weight
        self.rr_clipping = rr_clipping
        self.model_type = model_type
        self.regrate_rrrg = regrate_rrrg
        self.regrate_cdep = regrate_cdep

    def forward(self, model, X, y, expl, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_rrrg
        RRRG = RRRGradCamLoss.forward(self, model, X, y, expl.float(), logits, device)
        right_answer_loss += RRRG[1]
        right_reason_loss += RRRG[2]

        self.regularizer_rate = self.regrate_cdep
        CDEP = CDEPLoss.forward(self, model, X, y, expl.float(), logits, device)
        right_answer_loss += CDEP[1]
        right_reason_loss += CDEP[2][0]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss7(nn.Module):
    """
        MixLoss7 = CDEP + HINT
    """
    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, reduction='sum', model_type='mnist',\
        last_conv_specified=False, weight=None, rr_clipping=None, upsample=False, positive_only=False, regrate_cdep = 1000000, regrate_hint = 100):

        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping # good rate for decoy mnist 1.0
        self.weight = weight
        self.last_conv_specified = last_conv_specified
        self.reduction = reduction
        self.upsample = upsample
        self.model_type = model_type
        self.positive_only = positive_only
        self.regrate_cdep = regrate_cdep
        self.regrate_hint = regrate_hint

    def forward(self, model, X, y, expl_p, expl_r, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_cdep
        CDEP = CDEPLoss.forward(self, model, X, y, expl_p, logits, device)
        right_answer_loss += CDEP[1]
        right_reason_loss += CDEP[2][0]

        self.regularizer_rate = self.regrate_hint
        self.last_conv_specified = True
        self.upsample = True
        HINT = HINTLoss.forward(self, model, X, y, expl_r, logits, device)
        right_answer_loss += HINT[1]
        right_reason_loss += HINT[2][0]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss8(nn.Module):
    """
        MixLoss8 = RRR + HINT
    """
    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, reduction='sum',\
        last_conv_specified=False, weight=None, rr_clipping=None, upsample=False, positive_only=False, regrate_rrr = 10, regrate_hint = 100):
        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping
        self.weight = weight
        self.last_conv_specified = last_conv_specified
        self.reduction = reduction
        self.upsample = upsample
        self.positive_only = positive_only
        self.regrate_rrr = regrate_rrr
        self.regrate_hint = regrate_hint

    def forward(self, model, X, y, expl_p, expl_r, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_rrr
        RRR = RRRLoss.forward(self, X, y, expl_p, logits)
        right_answer_loss += RRR[1]
        right_reason_loss += RRR[2]

        self.regularizer_rate = self.regrate_hint
        self.last_conv_specified = True
        self.upsample = True
        HINT = HINTLoss.forward(self, model, X, y, expl_r, logits, device)
        right_answer_loss += HINT[1]
        right_reason_loss += HINT[2][0]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss8_ext(nn.Module):
    """
    MixLoss8_ext = RRR + HINT_IG
    """
    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, reduction='sum', \
                 weight=None, rr_clipping=None, regrate_rrr=10, regrate_hint_ig=100):
        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping
        self.weight = weight
        self.reduction = reduction
        self.regrate_rrr = regrate_rrr
        self.regrate_hint_ig = regrate_hint_ig

    def forward(self, model, X, y, expl_p, expl_r, logits, device, epoch, modelname):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_rrr
        RRR = RRRLoss.forward(self, X, y, expl_p, logits)
        right_answer_loss += RRR[1]
        right_reason_loss += RRR[2]

        self.regularizer_rate = self.regrate_hint_ig
        HINT_IG = HINTLoss_IG.forward(self, model, X, y, expl_r, logits, device)
        right_answer_loss += HINT_IG[1]
        right_reason_loss += HINT_IG[2][0]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss9(nn.Module):
    """
        MixLoss9 = RBR + HINT
    """
    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, reduction='sum',\
        last_conv_specified=False, weight=None, rr_clipping=None, upsample=False, positive_only=False, regrate_rbr = 100000, regrate_hint = 100):

        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping # good rate for decoy mnist 1.0
        self.weight = weight
        self.last_conv_specified = last_conv_specified
        self.reduction = reduction
        self.upsample = upsample
        self.positive_only = positive_only
        self.regrate_rbr = regrate_rbr
        self.regrate_hint = regrate_hint

    def forward(self, model, X, y, expl_p, expl_r, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_rbr
        RBR = RBRLoss.forward(self, model, X, y, expl_p, logits)
        right_answer_loss += RBR[1]
        right_reason_loss += RBR[2]

        self.regularizer_rate = self.regrate_hint
        self.last_conv_specified = True
        self.upsample = True
        HINT = HINTLoss.forward(self, model, X, y, expl_r, logits, device)
        right_answer_loss += HINT[1]
        right_reason_loss += HINT[2][0]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss11(nn.Module):
    """
    MixLoss11 = RBR + HINT_IG
    """
    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, reduction='sum',\
                 weight=None, rr_clipping=None, regrate_rbr=100000, regrate_hint_ig=100):
        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping  # good rate for decoy mnist 1.0
        self.weight = weight
        self.reduction = reduction
        self.regrate_rbr = regrate_rbr
        self.regrate_hint_ig = regrate_hint_ig

    def forward(self, model, X, y, expl_p, expl_r, logits, device, epoch, modelname):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_rbr
        RBR = RBRLoss.forward(self, model, X, y, expl_p, logits)
        right_answer_loss += RBR[1]
        right_reason_loss += RBR[2]

        self.regularizer_rate = self.regrate_hint_ig
        HINT_IG = HINTLoss_IG.forward(self, model, X, y, expl_r, logits, device)
        right_answer_loss += HINT_IG[1]
        right_reason_loss += HINT_IG[2][0]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss12(nn.Module):
    """
    MixLoss12 = CDEP + HINT_IG
    """

    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, weight=None, reduction='sum', \
                 model_type='mnist', rr_clipping=None, regrate_cdep = 1000000, regrate_hint_ig = 100):
        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.weight = weight
        self.reduction = reduction
        self.model_type = model_type
        self.rr_clipping = rr_clipping
        self.regrate_cdep = regrate_cdep
        self.regrate_hint_ig = regrate_hint_ig

    def forward(self, model, X, y, expl_p, expl_r, logits, device, epoch, modelname):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_cdep
        CDEP = CDEPLoss.forward(self, model, X, y, expl_p, logits, device)
        right_answer_loss += CDEP[1]
        right_reason_loss += CDEP[2][0]

        self.regularizer_rate = self.regrate_hint_ig
        HINT_IG = HINTLoss_IG.forward(self, model, X, y, expl_r, logits, device)
        right_answer_loss += HINT_IG[1]
        right_reason_loss += HINT_IG[2][0]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss13(nn.Module):
    """
    MixLoss13 = RRRG + HINT_IG
    """
    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, reduction='sum', \
                 last_conv_specified=False, weight=None, rr_clipping=None,
                 regrate_rrrg=1, regrate_hint_ig=100):
        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping  # good rate for decoy mnist 1.0
        self.weight = weight
        self.last_conv_specified = last_conv_specified
        self.reduction = reduction
        self.regrate_rrrg = regrate_rrrg
        self.regrate_hint_ig = regrate_hint_ig

    def forward(self, model, X, y, expl_p, expl_r, logits, device, epoch, modelname):
        right_answer_loss = 0
        right_reason_loss = 0

        self.regularizer_rate = self.regrate_rrrg
        RRRG = RRRGradCamLoss.forward(self, model, X, y, expl_p, logits, device)
        right_answer_loss += RRRG[1]
        right_reason_loss += RRRG[2]

        self.regularizer_rate = self.regrate_hint_ig
        HINT_IG = HINTLoss_IG.forward(self, model, X, y, expl_r, logits, device)
        right_answer_loss += HINT_IG[1]
        right_reason_loss += HINT_IG[2][0]

        right_answer_loss /= 2

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss14(nn.Module):
    """
    MixLoss14 = RRR + CE
    """
    def __init__(self, regularizer_rate=10, base_criterion=F.cross_entropy, weight=None, \
                 rr_clipping=None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing=0.0):
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
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.weight = weight
        self.rr_clipping = rr_clipping
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    # def forward(self, model, X, y, expl, mask, device):
    def forward(self, X, y, expl, mask, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        RRR = RRRLoss.forward(self, X, y, expl, logits, mask)
        right_answer_loss += RRR[1]
        right_reason_loss += RRR[2]

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss15(nn.Module):
    """
    MixLoss15 = RBR + CE
    """
    def __init__(self, regularizer_rate=1000000, base_criterion=F.cross_entropy, \
            rr_clipping=None, weight=None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing=0.0):
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
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.rr_clipping = rr_clipping # good rate for decoy mnist 1.0
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, model, X, y, expl, mask, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        RBR = RBRLoss.forward(self, model, X, y, expl, logits, mask)
        right_answer_loss += RBR[1]
        right_reason_loss += RBR[2]

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss16(nn.Module):
    """
    MixLoss16 = RRRG + CE
    """
    def __init__(self, regularizer_rate=1, base_criterion=F.cross_entropy, reduction='sum',\
        last_conv_specified=False, weight=None, rr_clipping=None, ignore_index: int = -100, label_smoothing=0.0):
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
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.reduction = reduction
        self.last_conv_specified = last_conv_specified
        self.weight = weight
        self.rr_clipping = rr_clipping
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, model, X, y, expl, mask, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0


        RRRG = RRRGradCamLoss.forward(self, model, X, y, expl, logits, device, mask)
        right_answer_loss += RRRG[1]
        right_reason_loss += RRRG[2]

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss17(nn.Module):
    """
    MixLoss17 = CDEP + CE
    """
    def __init__(self, regularizer_rate=1000, base_criterion=F.cross_entropy, weight=None, \
        model_type='mnist', rr_clipping=None, ignore_index: int = -100, reduction: str = 'mean', label_smoothing=0.0):
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
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.weight = weight
        self.model_type = model_type
        self.rr_clipping = rr_clipping
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, model, X, y, expl, mask, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        CDEP = CDEPLoss.forward(self, model, X, y, expl, logits, device, mask)
        right_answer_loss += CDEP[1]
        right_reason_loss += CDEP[2][0]

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLoss18(nn.Module):
    """
    MixLoss18 = HINT + CE
    """
    def __init__(self, regularizer_rate=0.1, base_criterion=F.cross_entropy, reduction='mean', \
        last_conv_specified=False, upsample=False, weight=None, positive_only=False, rr_clipping=None, \
                 ignore_index: int = -100, label_smoothing=0.0):
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
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.reduction = reduction
        self.last_conv_specified = last_conv_specified
        self.upsample = upsample
        self.weight = weight
        self.positive_only = positive_only
        self.rr_clipping = rr_clipping
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, model, X, y, expl, mask, logits, device):
        right_answer_loss = 0
        right_reason_loss = 0

        HINT = HINTLoss.forward(self, model, X, y, expl, logits, device, mask)
        right_answer_loss += HINT[1]
        right_reason_loss += HINT[2][0]

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss

class MixLossGeneral(nn.Module):
    """
    Combine Loss Function
    """

    def __init__(self, regularizer_rate=None, base_criterion=F.cross_entropy, weight=None, \
                 rr_clipping=None, reduction='sum', last_conv_specified=False, model_type='mnist', \
                 upsample=False, positive_only=False, regrate_rrr=None, regrate_rbr=None, \
                 regrate_rrrg=None, regrate_cdep=None, regrate_hint=None, regrate_hint_ig=None, ce=False):
        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.weight = weight
        self.rr_clipping = rr_clipping
        self.reduction = reduction
        self.last_conv_specified = last_conv_specified
        self.model_type = model_type
        self.upsample = upsample
        self.positive_only = positive_only
        self.regrate_rrr = regrate_rrr
        self.regrate_rbr = regrate_rbr
        self.regrate_rrrg = regrate_rrrg
        self.regrate_cdep = regrate_cdep
        self.regrate_hint = regrate_hint
        # self.regrate_hint_ig = regrate_hint_ig

    def forward(self, model, X, y, expl_p, expl_r, logits, device, mask=None):
        right_answer_loss = 0
        right_reason_loss = 0

        # breakpoint()
        right_answer_loss += self.base_criterion(logits, y)

        if self.regrate_rrr is not None:
            self.regularizer_rate = self.regrate_rrr
            RRR = RRRLoss.forward(self, X, y, expl_p, logits, mask)
            right_reason_loss += RRR[2]

        if self.regrate_rbr is not None:
            self.regularizer_rate = self.regrate_rbr
            RBR = RBRLoss.forward(self, model, X, y, expl_p.float(), logits, mask)
            right_reason_loss += RBR[2]

        if self.regrate_rrrg is not None:
            self.regularizer_rate = self.regrate_rrrg
            RRRG = RRRGradCamLoss.forward(self, model, X, y, expl_p.float(), logits, device, mask)
            right_reason_loss += RRRG[2]

        if self.regrate_cdep is not None:
            self.regularizer_rate = self.regrate_cdep
            CDEP = CDEPLoss.forward(self, model, X, y, expl_p.float(), logits, device, mask)
            right_reason_loss += CDEP[2][0]

        if self.regrate_hint is not None:
            self.regularizer_rate = self.regrate_hint
            self.last_conv_specified = True
            self.upsample = True
            HINT = HINTLoss.forward(self, model, X, y, expl_r, logits, device, mask)
            right_reason_loss += HINT[2][0]

        # if self.regrate_hint_ig is not None:
        #     self.regularizer_rate = self.regrate_hint_ig
        #     # self.last_conv_specified = True
        #     # self.upsample = True
        #     HINT_IG = HINTLoss_IG.forward(self, model, X, y, expl_r, logits, device
        #     right_reason_loss += HINT_IG[2][0]

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss


class MixLossGeneralRevised(nn.Module):
    """
    Revised version of MixLossGeneral
    """

    def __init__(
        self,
        regularizer_rate=None,          # STD
        base_criterion=F.cross_entropy, # STD (pytorch functional loss function. Is used in experiments which disable the XIL loss. Default cross_entropy.)
        weight=None,                    # UNUSED
        rr_clipping=None,               # STD
        reduction='sum',                # STD
        last_conv_specified=False,      # UNUSED ?!
        model_type='mnist',             # UNUSED
        upsample=False,                 # UNUSED ?!
        positive_only=False,            # UNUSED
        regrate_rrr=None,               # EXPL
        regrate_rbr=None,               # EXPL
        regrate_rrrg=None,              # EXPL
        regrate_cdep=None,              # EXPL
        regrate_hint=None,              # EXPL
        regrate_hint_ig=None,           # EXPL (commented)
        ce=False,
        explainers=[]
    ):

        super().__init__()
        self.regularizer_rate = regularizer_rate
        self.base_criterion = base_criterion
        self.weight = weight
        self.rr_clipping = rr_clipping
        self.reduction = reduction
        self.last_conv_specified = last_conv_specified
        self.model_type = model_type
        self.upsample = upsample
        self.positive_only = positive_only
        self.regrate_rrr = regrate_rrr
        self.regrate_rbr = regrate_rbr
        self.regrate_rrrg = regrate_rrrg
        self.regrate_cdep = regrate_cdep
        self.regrate_hint = regrate_hint
        # self.regrate_hint_ig = regrate_hint_ig


    def forward(self, model, X, y, expl_p, expl_r, logits, device, mask=None):
        right_answer_loss = 0
        right_reason_loss = 0

        # breakpoint()
        right_answer_loss += self.base_criterion(logits, y)

        if self.regrate_rrr is not None:
            RRR = self.regrate_rrr.forward(X, y, expl_p, logits, mask)
            right_reason_loss += RRR[2]

        if self.regrate_rbr is not None:
            RBR = self.regrate_rbr.forward(model, X, y, expl_p.float(), logits, mask)
            right_reason_loss += RBR[2]

        if self.regrate_rrrg is not None:
            RRRG = self.regrate_rrrg.forward(model, X, y, expl_p.float(), logits, device, mask)
            right_reason_loss += RRRG[2]

        if self.regrate_cdep is not None:
            CDEP = self.regrate_cdep.forward(model, X, y, expl_p.float(), logits, device, mask)
            right_reason_loss += CDEP[2][0]

        if self.regrate_hint is not None:
            HINT = self.regrate_hint.forward(model, X, y, expl_r, logits, device, mask)
            right_reason_loss += HINT[2][0]

        # if self.regrate_hint_ig is not None:
        #     self.regularizer_rate = self.regrate_hint_ig
        #     # self.last_conv_specified = True
        #     # self.upsample = True
        #     HINT_IG = HINTLoss_IG.forward(self, model, X, y, expl_r, logits, device
        #     right_reason_loss += HINT_IG[2][0]

        res = right_answer_loss + right_reason_loss

        return res, right_answer_loss, right_reason_loss



# not used
# def is_rawr(attribution, A, logits, y, threshold):
#     """
#     Return indices of instances in the batch which are Right Answer Wrong Reason (RAWR).

#     Args:
#         attribution: the unnormalized attribution of the explainer method (can be gradients, 
#             saliency maps). Tensor of shape (n, channels, h, w).
#         A: the ground-truth explanation feedback mask where ones indicate the confounding factor.
#         logits: the logits of the model.
#         y: the corresponding ground-truth labels to the logits.
#         threshold: the threshold defines the strictness (t=small —> strict, 
#             t=large —> not strict) with which the WR case is measured.
#     """
#     # check if right answer
#     preds = torch.max(F.softmax(logits, dim=1), 1)[1]
#     ra_mask = torch.eq(y, preds) # Trues for inputs that have right answer

#      # we need to clone the attribution because its called by reference and 
#      # we modify it in norm saliencies. We therefore detach it from the graph and clone it
#     attribution = attribution.detach().clone() 

#     # check if wrong reason
#     # we define wrong reason for one input as:
#     # -> sum of normalized attribution in the confounding region > threshold t 
#     # (in our case t=2, confounding region as indicated by the expl user mask; max value of sum
#     # =16 --> full focus on confoundfing factor, min=0 --> zero focus on confounding factor)

#     norm_attr = util.norm_saliencies_fast(attribution, positive_only=True)
#     attr_x_expl = torch.mul(A, norm_attr)
#     flat_attr_x_expl = attr_x_expl.view(attr_x_expl.size(0), -1)
#     thresholds = torch.sum(flat_attr_x_expl, dim=1)
#     wr_mask = torch.ge(thresholds, threshold)
#     ra_and_wr = torch.logical_and(ra_mask, wr_mask)
#     indices = torch.nonzero(ra_and_wr, as_tuple=True)[0]
#     how_many_rawr = len(indices) # save the number of instances that are RAWR
#     return indices, how_many_rawr
