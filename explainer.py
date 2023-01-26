"""Collection of explainer methods used to visualize/quantify explanations."""

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from captum.attr import LayerGradCam, LayerAttribution, Saliency, DeepLift,\
    InputXGradient, IntegratedGradients, GuidedBackprop, LRP
from lime import lime_image
from skimage.color import gray2rgb
from tqdm import tqdm
import json

import util


def explain_with_captum(method, model, dataloader, index_list, sign='positive', \
        fig_header='', next_to_each_other=False, save_name=None):
    """
    Explain with captum explainer. One grid with multiple images.

    Args:
        method: the explainer method to generate the explanations. The following 
            methods are available ['saliency', 'input_x_gradient', 'grad_cam', 
             'integrated_gradients'].
        model: the pytorch model.
        dataloader: pytorch dataloader (images, labels, _ ).
        index_list: list of indices for the images you want to explain. Takes the specified
            indices from the first batch of the dataloader (batchsize needs to be bigger 
            than the length of the specified indices.
        sign: specifies which attributions should be visualized. Either 'positive', 'negative',
            'absolute_value', 'all'. (see captum library viz.visualize_image_attr()).
        next_to_each_other: if True then plots one figure with the original image left and the 
            heatmap on the right side. If False then plot two seperate plots for org. images
            and corresponding heatmaps.
        save_name: if specified the resulting plot will be saved to output_images folder and not
            be printed. 
    Returns:
        Visualizes the images and the corresponding explanation heatmaps to two or one 
            plt figures (best used in Notebooks).

    """
    if fig_header == '':
        fig_header = method

    imgs, masks, titles = [], [], []

    model.eval()


    images_t, labels_t = util.load_first_batch(dataloader)
 
    for i in index_list:
        
        original_image = np.transpose((images_t[i].cpu().detach().numpy()), (1,2,0))
        img = images_t[i].unsqueeze(0)
        img.requires_grad_()
        gt = labels_t[i]
        output = model(img)

        _, predicted = torch.max(F.softmax(output, dim=1), 1)

        if method == 'saliency':
            saliency = Saliency(model)
            attr = saliency.attribute(img, target=predicted.item())

        elif method == 'input_x_gradient':
            ixg = InputXGradient(model)
            attr = ixg.attribute(img, target=predicted.item())
        
        elif method == 'grad_cam':
            model.zero_grad()
            last_conv_layer = util.get_last_conv_layer(model)
            grad_cam = LayerGradCam(model, last_conv_layer)
            attr = grad_cam.attribute(img, target=predicted.item(), relu_attributions=True)
            h, w = img.shape[2], img.shape[3]
            attr = LayerAttribution.interpolate(attr, (h, w))

        elif method == 'deep_lift':
            dl = DeepLift(model)
            attr = dl.attribute(img, target=predicted.item())

        elif method == 'lrp':
            lrp = LRP(model)
            attr = lrp.attribute(img, target=predicted.item())

        elif method == 'guided_backprop':
            gbp = GuidedBackprop(model)
            attr = gbp.attribute(img, target=predicted.item())
       
        elif method == 'integrated_gradients':
            intgrad = IntegratedGradients(model)
            attr = intgrad.attribute(img, target=predicted.item())

        elif method == 'kernel_shap':
            raise NotImplementedError("Currently kernel_shap is not working properly.")
            # TODO Lime not working 'ZeroDivisionError: Weights sum to zero, can't be normalized'
            #ks = KernelShap(model)
            #attr = ks.attribute(img, target=predicted.item(), n_samples=200)

        elif method == 'lime':
             raise NotImplementedError("Currently lime is not working properly.")
            # TODO: Often the attr is all zeros (can not be attributed); 
            # if 4x4 feature masks then zero all the time, even when increasing n_samples to 20000
            # don't know why this is happening???

            # lime = Lime(model)
            # feature_mask = util.generate_lime_feature_mask_8x8()
            # attr = lime.attribute(inputs= img, target=predicted.item(), \
            #     feature_mask=feature_mask, n_samples=2000)
            # if torch.sum(attr) == 0.:
            #     continue
        
        else:
            raise NotImplementedError

        # ... add more captum attribution methdos

        if attr.shape[1] == 1: # image with one channel (black white image)
            attr = attr.squeeze().unsqueeze(2).cpu().detach().numpy()
        else: # color image
            attr = np.transpose(attr.squeeze().detach().cpu().numpy(), (1,2,0))

        title = "y=" + str(gt.item()) + " pred=" + str(predicted.item())

        # store label, prediction
        if np.sum(attr) == 0.:
            print(f"Attribution mask is zero. Could not generate attribution with {method} for image at index {i}.")
            continue

        titles.append(title)
        imgs.append(original_image)
        masks.append(attr)

    if next_to_each_other:
        util.show_explanation_overlay_grid_captum_2(imgs, masks, titles, \
            sign=sign, fig_header=fig_header, method='heat_map', save_name=save_name)
    else:
        util.show_img_grid_numpy(imgs, save_name=save_name)
        util.show_explanation_overlay_grid_captum(imgs, masks, titles, sign=sign, \
            fig_header=fig_header, method='heat_map', save_name=save_name)

def explain_with_ig(model, dataloader, index_list, sign='positive', \
    fig_header='IG (Ross)', next_to_each_other=False, save_name=None):
    """
    Explain (visualize) the predictions of the model using Input Gradients 
    according to Ross et. al (2017). One grid with multiple images.

    Args:
        model: the trained pytorch model
        dataloader: torch dataloader with stored batches (X, y, expl), expl is optional
        index_list: list of indices for the images you want to explain. Takes the specified
            indices from the first batch of the dataloader (batchsize needs to be bigger 
            than the length of the specified indices.
        sign: specifies which attributions should be visualized. Either 'positive', 'negative',
            'absolute_value', 'all'. (see captum library viz.visualize_image_attr()).
        next_to_each_other: if True then plots one figure with the original image left and the 
            heatmap on the right side. If False then plot two separate plots for org. images
            and corresponding heatmaps.
        save_name: if specified the resulting plot will be saved to output_images folder and not
            be printed.  
    Returns:
        Visualizes the images and the corresponding explanation heatmaps to two 
            plt figures (best used in Notebooks).   
    """
    imgs, masks, titles = [], [], []

    model.eval()
    images_t, labels_t = util.load_first_batch(dataloader)
 
    for i in index_list:

        model.zero_grad()
        original_image = np.transpose((images_t[i].cpu().detach().numpy()), (1,2,0))
        img = images_t[i].unsqueeze(0)
        gt = labels_t[i]
        img.requires_grad_()

        logit = model(img)

        _, predicted = torch.max(F.softmax(logit, dim=1), 1)

        log_prob_ys = F.log_softmax(logit, dim=1)
        log_prob_ys.retain_grad()
        ig = torch.autograd.grad(log_prob_ys, img, torch.ones_like(log_prob_ys), \
            create_graph=True, allow_unused=True)[0]
    

        if ig.shape[1] == 1: # image with one channel (black white image)
            attr = ig.squeeze().unsqueeze(2).cpu().detach().numpy()
        else: # color image
            attr = np.transpose(ig.squeeze().detach().cpu().numpy(), (1,2,0))

        title = "y=" + str(gt.item()) + " pred=" + str(predicted.item())

        # store label, prediction
        titles.append(title)
        imgs.append(original_image)
        masks.append(attr)

    if next_to_each_other:
        util.show_explanation_overlay_grid_captum_2(imgs, masks, titles, sign=sign, \
            fig_header=fig_header, method='heat_map', save_name=save_name)

    else:
        util.show_img_grid_numpy(imgs, save_name=save_name)
        util.show_explanation_overlay_grid_captum(imgs, masks, titles, \
            sign=sign, fig_header='IG_Ross', method='heat_map', save_name=save_name)

def explain_with_lime(model, dataloader, index_list, sign='positive', \
    fig_header='Lime', next_to_each_other=False, gray_images=True, save_name=None):
    """
    Explain with Lime explainer. One grid with multiple images.

    Args:
        model: the pytorch model.
        dataloader: pytorch dataloader (images, labels, _ ).
        index_list: list of indices for the images you want to explain. Takes the specified
            indices from the first batch of the dataloader (batchsize needs to be bigger 
            than the lenght of the specified indices.
        sign: specifies which attributions should be visualized. Either 'positive', 'negative',
            'absolute_value', 'all'. (see captum library viz.visualize_image_attr()).
        next_to_each_other: if True then plots one figure with the original image left and the 
            heatmap on the right side. If False then plot two seperate plots for org. images
            and corresponding heatmaps. 
    Returns:
        Visualizes the images and the corresponding explanation heatmaps to two 
            plt figures (best used in Notebooks).

    """
    imgs, masks, titles = [], [], []
    images_t, labels_t = util.load_first_batch(dataloader)
    model.eval()

    for i in index_list:
        original_image = np.transpose((images_t[i].cpu().detach().numpy()), (1,2,0))
        img_numpy = images_t[i].squeeze().cpu().detach().numpy()
        if gray_images:
            img_numpy = gray2rgb(img_numpy).astype(np.double)
        else:
            img_numpy = np.transpose(img_numpy, (1,2,0)).astype(np.double)

        img_t = images_t[i].unsqueeze(0)
        gt = labels_t[i]
        output = model(img_t)

        _, predicted = torch.max(F.softmax(output, dim=1), 1)

        ######### helper functions for the LimeExplainer

        def get_preprocess_transform():
            """Convert the numpy images to the correct inputs for the model."""
            if gray_images:
                transf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ConvertImageDtype(dtype=torch.float32)
                ])
            else:
                transf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(dtype=torch.float32)
                ])

            return transf  
        preprocess_transform = get_preprocess_transform()

        def batch_predict(images):
            """Used to predict a batch of numpy images with the torch model in Lime."""
            model.eval()
            batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()

        #############

        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(img_numpy,    
                                            batch_predict, # classification function
                                            top_labels=1, 
                                            hide_color=0, 
                                            num_samples=1000) # number of images that will be sent to classification function
        
        #Map each explanation weight to the corresponding superpixel
        ind =  explanation.top_labels[0]
        dict_heatmap = dict(explanation.local_exp[ind])
        mask = np.expand_dims(np.vectorize(dict_heatmap.get)(explanation.segments), axis=2)
        
        title = "y=" + str(gt.item()) + " pred=" + str(predicted.item())
        # store label, prediction
        titles.append(title)
        imgs.append(original_image)
        masks.append(mask)
    
    if next_to_each_other:
        util.show_explanation_overlay_grid_captum_2(imgs, masks, titles, sign=sign, \
            fig_header=fig_header, method='heat_map', save_name=save_name)

    else:
        util.show_img_grid_numpy(imgs, save_name=save_name)
        util.show_explanation_overlay_grid_captum(imgs, masks, titles, sign=sign, \
            fig_header=fig_header, method='heat_map', save_name=save_name)


def explain_with_captum_one_by_one(method, model, dataloader, sign='positive', \
        fig_header='', next_to_each_other=False, save_name=None, clip_pixel_values_to_zero_one=True, device='cuda',
        specified_img_indices=[], flags=True):
    """
    Explain with captum explainer one plot per image in the dataloader.

    Args:
        method: the explainer method to generate the explanations. The following 
            methods are available ['saliency', 'input_x_gradient', 'grad_cam', 
             'integrated_gradients'].
        model: the pytorch model.
        dataloader: pytorch dataloader (images, labels, _ ).
        sign: specifies which attributions should be visualized. Either 'positive', 'negative',
            'absolute_value', 'all'. (see captum library viz.visualize_image_attr()).
        fig_header: title of the generated image.
        next_to_each_other: if True then plots one figure with the original image left and the 
            heatmap on the right side. If False then plot two seperate plots for org. images
            and corresponding heatmaps.
        save_name: if specified the resulting plot will be saved to output_images folder and not
            be printed.
        clip_pixel_values_to_zero_one: Squeeze pixels values to range [0,1] (used for RGB images)
        device: either 'cpu' or 'cuda' 
        specified_img_indices: list of indices for the images you want to explain. Takes the specified
            indices from the first batch of the dataloader (batchsize needs to be bigger 
            than the lenght of the specified indices.
    Returns:
        Visualizes the images and the corresponding explanation heatmaps to two or one 
            plt figures (best used in Notebooks).

    """
    if fig_header == '':
        fig_header = method
    model.eval()
    img_id = 0

    for data in dataloader:
        if flags:
            images_t, labels_t, mask_t, flags_t = data[0], data[1], data[2], data[3]
        else:
            images_t, labels_t, mask_t = data[0], data[1], data[2]
            flags_t = torch.ones(len(images_t))
 
        for i in range(len(labels_t)):

            if (len(specified_img_indices) == 0 or img_id in specified_img_indices) and flags_t[i] ==1:
                #util.show_image_tensor(mask_t[i], (299,299), save_name + '-mask')
                original_image = np.transpose((images_t[i].cpu().detach().numpy()), (1,2,0))
                img = images_t[i].unsqueeze(0).to(device)
                img.requires_grad_()
                gt = labels_t[i]
                output = model(img)

                _, predicted = torch.max(F.softmax(output, dim=1), 1)

                if method == 'saliency':
                    saliency = Saliency(model)
                    attr = saliency.attribute(img, target=predicted.item())

                elif method == 'input_x_gradient':
                    ixg = InputXGradient(model)
                    attr = ixg.attribute(img, target=predicted.item())
                
                elif method == 'grad_cam':
                    model.zero_grad()
                    last_conv_layer = util.get_last_conv_layer(model)
                    grad_cam = LayerGradCam(model, last_conv_layer)
                    attr = grad_cam.attribute(img, target=predicted.item(), relu_attributions=True)
                    h, w = img.shape[2], img.shape[3]
                    attr = LayerAttribution.interpolate(attr, (h, w))
            
                elif method == 'integrated_gradients':
                    intgrad = IntegratedGradients(model)
                    attr = intgrad.attribute(img, target=predicted.item())

                elif method == 'kernel_shap':
                    raise NotImplementedError("Currently kernel_shap is not working properly.")
                    # TODO Lime not working 'ZeroDivisionError: Weights sum to zero, can't be normalized'
                    #ks = KernelShap(model)
                    #attr = ks.attribute(img, target=predicted.item(), n_samples=200)

                elif method == 'lime':
                    raise NotImplementedError("Currently lime is not working properly.")
                    # TODO: Often the attr is all zeros (can not be attributed); 
                    # if 4x4 feature masks then zero all the time, even when increasing n_samples to 20000
                    # don't know why this is happening???

                    # lime = Lime(model)
                    # feature_mask = util.generate_lime_feature_mask_8x8()
                    # attr = lime.attribute(inputs= img, target=predicted.item(), \
                    #     feature_mask=feature_mask, n_samples=2000)
                    # if torch.sum(attr) == 0.:
                    #     continue
                
                else:
                    raise NotImplementedError

                # ... add more captum attribution methdos

                if attr.shape[1] == 1: # image with one channel (black white image)
                    attr = attr.squeeze().unsqueeze(2).cpu().detach().numpy()
                else: # color image
                    attr = np.transpose(attr.squeeze().detach().cpu().numpy(), (1,2,0))

                title = "y=" + str(gt.item()) + " pred=" + str(predicted.item())

                # store label, prediction
                if np.sum(attr) == 0.:
                    print(f"Attribution mask is zero. Could not generate attribution with {method} for image at index {i}.")
                    img_id +=1
                    continue

                if clip_pixel_values_to_zero_one:
                    original_image = np.clip(original_image, 0, 1)

                imgs, attrs, titles = [original_image], [attr], [title]

                cur_save_name = save_name + '-' +str(img_id)

                if next_to_each_other:
                    util.show_explanation_single(imgs, attrs, method='heat_map', sign='positive', \
                                                 save_name=cur_save_name)
                    #util.show_explanation_overlay_grid_captum_2(imgs, attrs, titles, sign=sign, \
                    #    fig_header=fig_header, method='heat_map', save_name=cur_save_name, figsize=(6,4))
                
                else:
                    util.show_img_grid_numpy(imgs, save_name=save_name)
                    util.show_explanation_overlay_grid_captum(imgs, attrs, titles, sign=sign, \
                        fig_header=fig_header, method='heat_map', save_name=cur_save_name)
                print(f"explanation image with name {cur_save_name} saved!")
                
                img_id += 1
            else:
                img_id +=1
                continue

def explain_with_lime_one_by_one(model, dataloader, sign='positive', \
    fig_header='Lime', next_to_each_other=False, gray_images=False, save_name=None, device='cuda', \
        clip_pixel_values_to_zero_one=True, specified_img_indices=[], num_samples=1000):
    """
    Explain with lime explainer one plot per image in the dataloader.

    Args:
        model: the pytorch model.
        dataloader: pytorch dataloader (images, labels, _ ).
        sign: specifies which attributions should be visualized. Either 'positive', 'negative',
            'absolute_value', 'all'. (see captum library viz.visualize_image_attr()).
        fig_header: title of the generated image.
        next_to_each_other: if True then plots one figure with the original image left and the 
            heatmap on the right side. If False then plot two seperate plots for org. images
            and corresponding heatmaps.
        gray_images: set to True if input are gray images (MNIST)
        save_name: if specified the resulting plot will be saved to output_images folder and not
            be printed.
        clip_pixel_values_to_zero_one: Squeeze pixels values to range [0,1] (used for RGB images)
        device: either 'cpu' or 'cuda' 
        specified_img_indices: list of indices for the images you want to explain. Takes the specified
            indices from the first batch of the dataloader (batchsize needs to be bigger 
            than the lenght of the specified indices.
        num_samples: number of samples LIME explainer should use (see LIME library for details)
    Returns:
        Visualizes the images and the corresponding explanation heatmaps to two or one 
            plt figures (best used in Notebooks).

    """
    model.eval()
    img_id = 0

    for data in dataloader:
        images_t, labels_t = data[0], data[1]
        for i in range(len(labels_t)):
            if len(specified_img_indices) == 0 or img_id in specified_img_indices:

                original_image = np.transpose((images_t[i].cpu().detach().numpy()), (1,2,0))
                img_numpy = images_t[i].squeeze().cpu().detach().numpy()
                if gray_images:
                    img_numpy = gray2rgb(img_numpy).astype(np.double)
                else:
                    img_numpy = np.transpose(img_numpy, (1,2,0)).astype(np.double)

                img_t = images_t[i].unsqueeze(0).to(device)
                gt = labels_t[i]
                output = model(img_t)

                _, predicted = torch.max(F.softmax(output, dim=1), 1)

                ######### helper functions for the LimeExplainer

                def get_preprocess_transform():
                    """Convert the numpy images to the correct inputs for the model."""
                    if gray_images:
                        transf = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ConvertImageDtype(dtype=torch.float32)
                        ])
                    else:
                        transf = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.ConvertImageDtype(dtype=torch.float32)
                        ])

                    return transf  
                preprocess_transform = get_preprocess_transform()

                def batch_predict(images):
                    """Used to predict a batch of numpy images with the torch model in Lime."""
                    model.eval()
                    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model.to(device)
                    batch = batch.to(device)
                    logits = model(batch)
                    probs = F.softmax(logits, dim=1)
                    return probs.detach().cpu().numpy()

                #############

                explainer = lime_image.LimeImageExplainer()
                explanation = explainer.explain_instance(img_numpy,    
                                                    batch_predict, # classification function
                                                    top_labels=1, 
                                                    hide_color=0, 
                                                    num_samples=num_samples) # number of images that will be sent to classification function
                
                #Map each explanation weight to the corresponding superpixel
                ind =  explanation.top_labels[0]
                dict_heatmap = dict(explanation.local_exp[ind])
                attr = np.expand_dims(np.vectorize(dict_heatmap.get)(explanation.segments), axis=2)
                
                title = "y=" + str(gt.item()) + " pred=" + str(predicted.item())
                # store label, prediction

                if np.sum(attr) == 0.:
                    print(f"Attribution mask is zero. Could not generate attribution with lime for image at index {i}.")
                    img_id +=1
                    continue

                if clip_pixel_values_to_zero_one:
                    original_image = np.clip(original_image, 0, 1)

                imgs, attrs, titles = [original_image], [attr], [title]

                cur_save_name = save_name + '-' +str(img_id)
            
                if next_to_each_other:
                    util.show_explanation_single(imgs, attrs, method='heat_map', sign='positive', \
                                                 save_name=cur_save_name)
                    #util.show_explanation_overlay_grid_captum_2(imgs, attrs, titles, sign=sign, \
                    #    fig_header=fig_header, method='heat_map', save_name=cur_save_name, figsize=(6,4))

                else:
                    util.show_img_grid_numpy(imgs, save_name=cur_save_name)
                    util.show_explanation_overlay_grid_captum(imgs, attrs, titles, sign=sign, \
                        fig_header=fig_header, method='heat_map', save_name=cur_save_name)

                print(f"explanation image with name {cur_save_name} saved!")
                        
                img_id += 1
                
            else:
                img_id +=1
                continue


def explain_with_ig_one_by_one(model, dataloader, sign='positive', \
    fig_header='IG (Ross)', next_to_each_other=False, save_name=None, device='cuda', \
        clip_pixel_values_to_zero_one=True, specified_img_indices=[]):
    """
    Explain (visualize) the predictions of the model using Input Gradients 
    according to Ross et. al (2017). One image per plot.
    Args:
        model: the pytorch model.
        dataloader: pytorch dataloader (images, labels, _ ).
        sign: specifies which attributions should be visualized. Either 'positive', 'negative',
            'absolute_value', 'all'. (see captum library viz.visualize_image_attr()).
        fig_header: title of the generated image.
        next_to_each_other: if True then plots one figure with the original image left and the 
            heatmap on the right side. If False then plot two seperate plots for org. images
            and corresponding heatmaps.
        gray_images: set to True if input are gray images (MNIST)
        save_name: if specified the resulting plot will be saved to output_images folder and not
            be printed.
        clip_pixel_values_to_zero_one: Squeeze pixels values to range [0,1] (used for RGB images)
        device: either 'cpu' or 'cuda' 
        specified_img_indices: list of indices for the images you want to explain. Takes the specified
            indices from the first batch of the dataloader (batchsize needs to be bigger 
            than the lenght of the specified indices.

    Returns:
        Visualizes the images and the corresponding explanation heatmaps to two or one 
            plt figures (best used in Notebooks).
    """
    model.eval()
    img_id = 0

    for data in dataloader:
        images_t, labels_t = data[0], data[1]
        for i in range(len(labels_t)):
            if len(specified_img_indices) == 0 or img_id in specified_img_indices:

                model.zero_grad()
                original_image = np.transpose((images_t[i].cpu().detach().numpy()), (1,2,0))
                img = images_t[i].unsqueeze(0).to(device)
                gt = labels_t[i]
                img.requires_grad_()

                logit = model(img)

                _, predicted = torch.max(F.softmax(logit, dim=1), 1)

                log_prob_ys = F.log_softmax(logit, dim=1)
                log_prob_ys.retain_grad()
                ig = torch.autograd.grad(log_prob_ys, img, torch.ones_like(log_prob_ys), \
                    create_graph=True, allow_unused=True)[0]
            

                if ig.shape[1] == 1: # image with one channel (black white image)
                    attr = ig.squeeze().unsqueeze(2).cpu().detach().numpy()
                else: # color image
                    attr = np.transpose(ig.squeeze().detach().cpu().numpy(), (1,2,0))

                title = "y=" + str(gt.item()) + " pred=" + str(predicted.item())

                if np.sum(attr) == 0.:
                    print(f"Attribution mask is zero. Could not generate attribution with ig for image at index {i}.")
                    img_id +=1
                    continue

                if clip_pixel_values_to_zero_one:
                    original_image = np.clip(original_image, 0, 1)

                imgs, attrs, titles = [original_image], [attr], [title]

                cur_save_name = save_name + '-' +str(img_id)
            
                if next_to_each_other:
                    util.show_explanation_single(imgs, attrs, method='heat_map', sign='positive', \
                                                 save_name=cur_save_name)
                    #util.show_explanation_overlay_grid_captum_2(imgs, attrs, titles, sign=sign, \
                    #    fig_header=fig_header, method='heat_map', save_name=cur_save_name, figsize=(6,4))

                else:
                    util.show_img_grid_numpy(imgs, save_name=cur_save_name)
                    util.show_explanation_overlay_grid_captum(imgs, attrs, titles, sign=sign, \
                        fig_header=fig_header, method='heat_map', save_name=cur_save_name)

                print(f"explanation image with name {cur_save_name} saved!")
                        
                img_id += 1

            else:
                img_id +=1
                continue

# ### WRONG REASON QUANTIFICATION: WR METRIC

def quantify_wrong_reason(method, dataloader, model, device, name, wr_name,\
    foldername="output_wr_metric/", threshold=None, mode='mean', flags=True):
    """
    Quantifies wrong reason based on ground-truth explanations (i.e.
    measures the confounder activation for whole test set). Either mean or
    median pixel-level activation. If threshold=None then calculate 
    the binarization threshold. To binarize heatmaps before measurement insert
    threshold.

    Args:
        method: either 'ig_ross' or 'grad_cam'
        dataloader: pytorch dataloader (images, _, expl, flags) where expl are the 
            ground-truth masks.
        model: the pytorch model.
        device: either 'cpu' or 'cuda'
        name: name of the model or identifier
        foldername: name of the output folder 
        threshold: if None, then calculates the mean or median avg activation 
            binarization threshold.
        mode: either 'mean' or 'median'. Mean calculates the mean pixel-level activation
            averaged over the whole dataloader. Median calculates the median pixel-level
            activation and returns the median of the whole dataloader.
        flags: it True then only images with flag=1 are taking into account (ISIC19).
 
    """

    model.eval() 
    actScores = []
    number_instances = 0
    cannot_attribute_num = 0
    img_num = []

    with tqdm(dataloader, unit="batch") as tbatch:
        for num, data in enumerate(tbatch):
            if flags:
                images_t, masks_t, flags_t = data[0].to(device), data[2].to(device), data[3].to(device)
            else:
                images_t, masks_t = data[0].to(device), data[2].to(device)

            images_t.requires_grad_()
            logits = model(images_t)
            h, w = images_t.shape[2], images_t.shape[3]
            _, predicted = torch.max(F.softmax(logits, dim=1), 1)

            if method == 'grad_cam':
                # _, predicted = torch.max(F.softmax(logits, dim=1), 1)
                # network importance score --> compute GradCam attribution of last conv layer
                last_conv_layer = util.get_last_conv_layer(model)
                explainer = LayerGradCam(model, last_conv_layer)
                attr = explainer.attribute(images_t, target=predicted, relu_attributions=True)
                # upsample attr
                attr = LayerAttribution.interpolate(attr, (h, w))
                # attr.shape => (n,1,h,w)

            elif method == 'saliency':
                sal = Saliency(model)
                attr = sal.attribute(images_t, target=predicted)

            elif method == 'input_x_gradient':
                ixg = InputXGradient(model)
                attr = ixg.attribute(images_t, target=predicted)

            elif method == 'deep_lift':
                dl = DeepLift(model)
                attr = dl.attribute(images_t, target=predicted)

            elif method == 'guided_backprop':
                gbp = GuidedBackprop(model)
                attr = gbp.attribute(images_t, target=predicted)

            elif method == 'lrp':
                lrp = LRP(model)
                attr = lrp.attribute(images_t, target=predicted)

            elif method == 'integrated_gradient':
                intgrad = IntegratedGradients(model)
                attr = intgrad.attribute(images_t, target=predicted)

            elif method == 'ig_ross':
                model.zero_grad()
                # get gradients w.r.t. to the input
                log_prob_ys = F.log_softmax(logits, dim=1)
                log_prob_ys.retain_grad()
                attr = torch.autograd.grad(log_prob_ys, images_t, torch.ones_like(log_prob_ys), \
                    create_graph=True, allow_unused=True)[0]
                # attr.shape => (n,3,h,w)
                # sum over rgb channels
                attr = torch.sum(attr, dim=1).unsqueeze(1)

            
            norm_attr = util.norm_saliencies_fast(attr, positive_only=True)
            # filter out instances that have no seg mask
            if flags:
                have_confounder_indices = torch.nonzero((flags_t == 1), as_tuple=True)[0]
                norm_attr = norm_attr[have_confounder_indices]
                masks_t = masks_t[have_confounder_indices]
                tmp = len(have_confounder_indices)
            else:
                tmp = len(images_t)
            # filter out attr which are zero --> assuming no attr can be calculated
            sums = torch.sum(norm_attr, dim=(1,2,3))
            only_non_zero_attr_indices = torch.nonzero(sums, as_tuple=True)[0]
            norm_attr = norm_attr[only_non_zero_attr_indices]
            masks_t = masks_t[only_non_zero_attr_indices]
            #flags_t = flags_t[only_non_zero_attr_indices]
            cannot_attribute_num += (tmp - len(only_non_zero_attr_indices))
            number_instances += len(only_non_zero_attr_indices)

            if norm_attr.size(0) == 0:
                continue 

            if threshold is not None:
                norm_attr[norm_attr > threshold] = 1.0
                norm_attr[norm_attr < threshold] = 0.0

                attr_x_expl = torch.mul(masks_t, norm_attr)
                flat_attr_x_expl = attr_x_expl.view(attr_x_expl.size(0), -1)
                attr_ca = torch.sum(flat_attr_x_expl, dim=1)
                masks_flat = masks_t.view(masks_t.size(0), -1)
                attr_max = torch.sum(masks_flat, dim=1)

                actScore = torch.div(attr_ca, attr_max)
                actScores += actScore.tolist()
                if dataloader.batch_size == 1:
                    img_num.append(num)
                # print("actScore " + str(actScore))

            else: # calc median/mean
                if mode == 'mean':
                    scores = torch.div(torch.sum(norm_attr.view(norm_attr.size(0), -1), dim=1), (h*w))
                elif mode == 'median':
                    scores = torch.median(norm_attr.view(norm_attr.size(0), -1), dim=1)[0]
                actScores += scores.tolist()
                if dataloader.batch_size == 1:
                    img_num.append(num)
            #     print("scores " + str(scores))

            # print(" !!!!!! SCORES = " + str(a))
            # check if list item in actScores equal to 'BATCH_SIZE'


    if number_instances != len(actScores):
        raise RuntimeWarning(f"Counted instances which have seg mask ({number_instances}) not equal to length of actScores ({len(actScores)})!!!")
    
    if threshold is not None:
        abs_activation = sum(actScores)
        avg_activation_per_instance = np.mean(actScores)
        std = np.std(actScores)
        #print(f"ActScores: {actScores}")
        print(f"Number of actScores= {len(actScores)}")
        print(f"Activation AVG per instance = {100*avg_activation_per_instance}")
        print(f"STD = {100*std}")
        print(f"Activation ABS sum = {abs_activation}")
        print(f"Number of complete zero attr= {cannot_attribute_num}")
        data = {'name': name,
                'avg_activation_per_instance': 100*avg_activation_per_instance,
                'std_activation_per_instance': 100*std,
                'abs_activation': abs_activation,
                'length': number_instances,
                'cannot_attribute_num': cannot_attribute_num,
                'actScores': actScores
            }
        with open(foldername + name + '-wrong_reason_stats.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        def takeSecond(elem):
            return elem[1]

        if img_num:
            arr = np.array(actScores)
            arr = 100 * arr
            actScores = arr.tolist()
            wr_score = list(zip(img_num, actScores))
            wr_score.sort(key=takeSecond, reverse=True)
            # breakpoint()
            wr_score = list(zip(img_num, wr_score))
            f = open(f"./img_wr_metric/{wr_name}.txt", "w")
            f.write(f'\t img_num \t wr_score \n')
            for i in wr_score:
                line = str(i[0]) + "\t" + str(i[1][0]) + "\t\t\t" + str(i[1][1])
                f.write(f'{line}\n')
            f.close()

        # breakpoint()

        return 100*avg_activation_per_instance

    else:
        if mode == 'mean':
            value = np.mean(actScores)
        elif mode == 'median':
            value = np.median(actScores) # get median of list of medians

        print(f"{str(mode)}= {float(value)}")

        print(f"Number of actScores= {len(actScores)}")
        print(f"Number of complete zero attr= {cannot_attribute_num}")
        data = {'name': name,
                 str(mode): float(value),
                'num': number_instances,
                'cannot_attribute_num' : cannot_attribute_num,
                'actScores': actScores
            }
        
        with open(foldername + name + '-' + str(mode) + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # if img_num:
        #     arr = np.array(actScores)
        #     arr = 100 * arr
        #     actScores = arr.tolist()
        #     wr_score = list(zip(img_num, actScores))

        return float(value)#, len(actScores)

def quantify_wrong_reason_lime(dataloader, model, name, foldername="output_wr_metric/",\
     threshold=None, mode='mean', gray_images=False, num_samples=1000, save_raw_attr=False, flags=True):
    """
    Quantifies LIME wrong reason based on ground-truth explanations (i.e.
    measures the confounder activation for whole test set). Either mean or
    median pixel-level activation. If threshold=None then calculate 
    the binarization threshold. To binarize heatmaps before measurement insert
    threshold.

    Args:
        dataloader: pytorch dataloader (images, _, expl, flags) where expl are the 
            ground-truth masks.
        model: the pytorch model.
        device: either 'cpu' or 'cuda'
        name: name of the model or identifier
        foldername: name of the output folder 
        threshold: if None, then calculates the mean or median avg activation 
            binarization threshold.
        mode: either 'mean' or 'median'. Mean calculates the mean pixel-level activation
            averaged over the whole dataloader. Median calculates the median pixel-level
            activation and returns the median of the whole dataloader.
        gray_images: set to True if input images are black and white (MNIST)
        num_samples: number of samples LIME explainer should use (see LIME library for details)
        save_raw_attr: if True saves the unnormalized attributions to the folder specified
            in foldername with the given 'name'. 
        flags: it True then only images with flag=1 are taking into account (ISIC19).
 
    """
    
    model.eval()
    actScores = []
    number_instances = 0
    cannot_attribute_num = 0

    if save_raw_attr:
        attr_store = []
        mask_store = []
    
    with tqdm(dataloader, unit="batch") as tbatch:
        for data in tbatch:
            if flags:
                images_t, masks_t, flags_t = data[0], data[2], data[3]
            else:
                images_t, masks_t = data[0], data[2]
                flags_t = torch.ones(len(images_t))
            h, w = images_t.shape[2], images_t.shape[3]

            for i in range(len(flags_t)): # LimeExplainer needs numpy arrays, we cannot use GPU
                if flags_t[i].item() == 1:

                    img_numpy = images_t[i].squeeze().cpu().detach().numpy()
                    if gray_images:
                        img_numpy = gray2rgb(img_numpy).astype(np.double)
                    else:
                        img_numpy = np.transpose(img_numpy, (1,2,0)).astype(np.double)

                    ######### helper functions for the LimeExplainer

                    def get_preprocess_transform():
                        """Convert the numpy images to the correct inputs for the model."""
                        if gray_images:
                            transf = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ConvertImageDtype(dtype=torch.float32)
                            ])
                        else:
                            transf = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.ConvertImageDtype(dtype=torch.float32)
                            ])

                        return transf 
                     
                    preprocess_transform = get_preprocess_transform()

                    def batch_predict(images):
                        """Used to predict a batch of numpy images with the torch model in Lime."""
                        model.eval()
                        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model.to(device)
                        batch = batch.to(device)
                        logits = model(batch)
                        probs = F.softmax(logits, dim=1)
                        return probs.detach().cpu().numpy()

                    #############

                    explainer = lime_image.LimeImageExplainer()
                    explanation = explainer.explain_instance(img_numpy,    
                                                        batch_predict, # classification function
                                                        top_labels=1, 
                                                        hide_color=0, 
                                                        num_samples=num_samples) # number of images that will be sent to classification function
                    
                    #Map each explanation weight to the corresponding superpixel
                    ind =  explanation.top_labels[0]
                    dict_heatmap = dict(explanation.local_exp[ind])
                    attr_np = np.expand_dims(np.vectorize(dict_heatmap.get)(explanation.segments), axis=2)
                    attr_np = np.moveaxis(attr_np, -1, 0)
                    attr = torch.from_numpy(attr_np).unsqueeze(0).type(torch.FloatTensor) # attr.shape => (1,1,h,w)
                    
                    # store label, prediction
                    mask = masks_t[i].unsqueeze(0)
                    if save_raw_attr:
                        attr_store.append(attr)
                        mask_store.append(mask)

                    norm_attr = util.norm_saliencies_fast(attr, positive_only=True)
                    if torch.sum(norm_attr) == 0.:
                        print(f"Attribution mask is zero. Could not generate attribution with lime for image at index {i}.")
                        cannot_attribute_num += 1
                        continue

                    number_instances += 1

                    if threshold is not None:
                        norm_attr[norm_attr > threshold] = 1.0
                        norm_attr[norm_attr < threshold] = 0.0

                        attr_x_expl = torch.mul(mask, norm_attr)
                        flat_attr_x_expl = attr_x_expl.view(attr_x_expl.size(0), -1)
                        attr_ca = torch.sum(flat_attr_x_expl, dim=1)
                        mask_flat = mask.view(mask.size(0), -1)
                        attr_max = torch.sum(mask_flat, dim=1)

                        actScore = torch.div(attr_ca, attr_max)
                        actScores += actScore.tolist()

                    else: # calc median/mean
                        if mode == 'mean':
                            scores = torch.div(torch.sum(norm_attr.view(norm_attr.size(0), -1), dim=1), (h*w))
                        elif mode == 'median':
                            scores = torch.median(norm_attr.view(norm_attr.size(0), -1), dim=1)[0]
                        actScores += scores.tolist()

        if number_instances != len(actScores):
            raise RuntimeWarning(f"Counted instances which have seg mask ({number_instances}) not equal to length of actScores ({len(actScores)})!!!")

        if save_raw_attr:
            store_attr = torch.cat(attr_store, dim=0).type(torch.FloatTensor)
            store_mask = torch.cat(mask_store, dim=0).type(torch.FloatTensor) 
            torch.save(store_attr, foldername + name + '-attr.pt')
            torch.save(store_mask, foldername + name + '-mask.pt')        
        
        if threshold is not None: # Calculate threshold
            abs_activation = sum(actScores)
            avg_activation_per_instance = np.mean(actScores)
            std = np.std(actScores)
            #print(f"ActScores: {actScores}")
            print(f"Number of actScores= {len(actScores)}")
            print(f"Activation AVG per instance = {100*avg_activation_per_instance}")
            print(f"STD = {100*std}")
            print(f"Activation ABS sum = {abs_activation}")
            print(f"Number of complete zero attr= {cannot_attribute_num}")
            data = {'name': name,
                    'avg_activation_per_instance': 100*avg_activation_per_instance,
                    'std_activation_per_instance': 100*std,
                    'abs_activation': abs_activation,
                    'length': number_instances,
                    'cannot_attribute_num': cannot_attribute_num,
                    'actScores': actScores
                }
            with open(foldername + name + str(mode) + '-wrong_reason_stats.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        else:
            if mode == 'mean':
                value = np.mean(actScores)
            elif mode == 'median':
                value = np.median(actScores) # get median of list of medians

            print(f"{str(mode)}= {float(value)}")

            print(f"Number of actScores= {len(actScores)}")
            print(f"Number of complete zero attr= {cannot_attribute_num}")
            data = {'name': name,
                    str(mode): float(value),
                    'num': number_instances,
                    'cannot_attribute_num' : cannot_attribute_num,
                    'actScores': actScores
                }
            
            with open(foldername + name + '-' + str(mode) + '.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        
        return value#, len(actScores)

def quantify_wrong_reason_lime_preload(model, name, foldername="output_wr_metric/",\
     threshold=None, mode='mean', device='cuda', batch_size=16):
    """
    Used to load precalculated LIME attributions and calculate WR.
    Use quantify_wrong_reason_lime to calculate attribution files.
    """
    
    model.eval()
    actScores = []
    number_instances = 0
    cannot_attribute_num = 0

    # load attr and masks
    attrs = torch.load(foldername + name + '-attr.pt')
    masks = torch.load(foldername + name + '-mask.pt')

    from torch.utils.data import DataLoader, TensorDataset
    t = TensorDataset(attrs, masks)
    dataloader = DataLoader(t, batch_size=batch_size)

    with tqdm(dataloader, unit="batch") as tbatch:
        for data in tbatch:
            attr, mask = data[0].to(device), data[1].to(device)
            h, w = attr.shape[2], attr.shape[3]
            batch_number = len(attr)

            norm_attr = util.norm_saliencies_fast(attr, positive_only=True)
            
            # filter out zero attributions
            sums = torch.sum(norm_attr, dim=(1,2,3))
            only_non_zero_attr_indices = torch.nonzero(sums, as_tuple=True)[0]
            norm_attr = norm_attr[only_non_zero_attr_indices]
            mask = mask[only_non_zero_attr_indices]
            cannot_attribute_num += (batch_number - len(only_non_zero_attr_indices))

            if norm_attr.size(0) == 0:
                continue 

            number_instances += len(mask)

            if threshold is not None:
                norm_attr[norm_attr > threshold] = 1.0
                norm_attr[norm_attr < threshold] = 0.0

                attr_x_expl = torch.mul(mask, norm_attr)
                flat_attr_x_expl = attr_x_expl.view(attr_x_expl.size(0), -1)
                attr_ca = torch.sum(flat_attr_x_expl, dim=1)
                mask_flat = mask.view(mask.size(0), -1)
                attr_max = torch.sum(mask_flat, dim=1)

                actScore = torch.div(attr_ca, attr_max)
                actScores += actScore.tolist()

            else: # calc median/mean
                if mode == 'mean':
                    scores = torch.div(torch.sum(norm_attr.view(norm_attr.size(0), -1), dim=1), (h*w))
                elif mode == 'median':
                    scores = torch.median(norm_attr.view(norm_attr.size(0), -1), dim=1)[0]
                actScores += scores.tolist()

        if number_instances != len(actScores):
            raise RuntimeWarning(f"Counted instances which have seg mask ({number_instances}) not equal to length of actScores ({len(actScores)})!!!")       
        
        if threshold is not None:
            abs_activation = sum(actScores)
            avg_activation_per_instance = np.mean(actScores)
            std = np.std(actScores)
            #print(f"ActScores: {actScores}")
            print(f"Number of actScores= {len(actScores)}")
            print(f"Activation AVG per instance = {100*avg_activation_per_instance}")
            print(f"STD = {100*std}")
            print(f"Activation ABS sum = {abs_activation}")
            print(f"Number of complete zero attr= {cannot_attribute_num}")
            data = {'name': name,
                    'avg_activation_per_instance': 100*avg_activation_per_instance,
                    'std_activation_per_instance': 100*std,
                    'abs_activation': abs_activation,
                    'length': number_instances,
                    'cannot_attribute_num': cannot_attribute_num,
                    'actScores': actScores
                }
            with open(foldername + name + '-wrong_reason_stats.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        else:
            if mode == 'mean':
                value = np.mean(actScores)
            elif mode == 'median':
                value = np.median(actScores) # get median of list of medians

            print(f"{str(mode)}= {float(value)}")

            print(f"Number of actScores= {len(actScores)}")
            print(f"Number of complete zero attr= {cannot_attribute_num}")
            data = {'name': name,
                    str(mode): float(value),
                    'num': number_instances,
                    'cannot_attribute_num' : cannot_attribute_num,
                    'actScores': actScores
                }
            
            with open(foldername + name + '-' + str(mode) + '.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        
        return 100*avg_activation_per_instance

def store_mean_quantification_score_per_image_confounder(dataloader, model, \
        device, foldername="output_wr_metric/", save_name='train'):
    """
    HELPER function: Stores indices of dataloader from most informative to least 
    based on the GradCAM WR activation.
    Used for the ISIC19 Interaction efficiancy experiment.  
    """    
    model.to(device)
    model.eval() 
    store = []

    with tqdm(dataloader, unit="batch") as tbatch:
        for data in tbatch:
            images_t, masks_t, flags_t = data[0].to(device), data[2].to(device), data[3].to(device)
            images_t.requires_grad_()
            logits = model(images_t)
            h, w = images_t.shape[2], images_t.shape[3]

        
            _, predicted = torch.max(F.softmax(logits, dim=1), 1)
            # network importance score --> compute GradCam attribution of last conv layer
            last_conv_layer = util.get_last_conv_layer(model)
            explainer = LayerGradCam(model, last_conv_layer)
            attr = explainer.attribute(images_t, target=predicted, relu_attributions=True)
            # upsample attr
            attr = LayerAttribution.interpolate(attr, (h, w))
            # attr.shape => (n,1,h,w)
            # zero out instances that have no seg mask
            have_not_confounder_indices = torch.nonzero((flags_t == 0), as_tuple=True)[0]
            norm_attr = util.norm_saliencies_fast(attr, positive_only=True)
            norm_attr[have_not_confounder_indices] = torch.zeros((1,h,w)).to(device)
        
            attr_x_expl = torch.mul(masks_t, norm_attr)
            flat_attr_x_expl = attr_x_expl.view(attr_x_expl.size(0), -1)
            attr_ca = torch.sum(flat_attr_x_expl, dim=1)
            masks_flat = masks_t.view(masks_t.size(0), -1)
            attr_max = torch.sum(masks_flat, dim=1) + 1e-12

            actScore = torch.div(attr_ca, attr_max)
            store += actScore.tolist()

    attr_values = np.array(store)
    sorted_attr_values_ind = np.argsort(attr_values)[::-1]
    with open(foldername + "informative_score_indices_"+ str(save_name) + "_set_most_to_least.npy", 'wb') as f:
        np.save(f, sorted_attr_values_ind)
    
    print(f"LENGTH OF ATTR VALUES: {len(attr_values)}")
    print(f"LENGTH OF SORTED IND: {len(sorted_attr_values_ind)}")
