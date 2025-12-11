import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig

from data_utils import load_images
from model_utils import WrappedModel, load_model
from position_utils import reshape_to_patches, patches_to_image, perturb_X, perturb_Xp, weight, RidgeRegression

def lime(X, model, process_fn=patches_to_image, position=True, nperturb=100, nepochs=100): 
    with torch.no_grad():
        outputs = model(process_fn(X).cuda())
    yhat = outputs.argmax(1)
    #print(yhat)
    X_lime_list, y_lime_list, weights_list = [], [], []
    for _ in range(nperturb):
        X0, indicator_X0 = perturb_X(X)
        if position:
            Xp0, indicator_Xp0 = perturb_Xp(X0)
        torch.cuda.empty_cache()
        with torch.no_grad():
            outputs0 = model(process_fn(Xp0 if position else X0).cuda())
        logit0 = outputs0[torch.arange(outputs0.size(0)), yhat]
        X_weights = weight(X0, X)
        if position:
            X_lime_sample = torch.cat((indicator_X0.to('cuda'), indicator_Xp0.to('cuda')), dim=1) # 1 x 392
            Xp_weights = weight(Xp0, X0)
            weights_sample = torch.cat((X_weights.to('cuda'), Xp_weights.to('cuda')), dim=1) # 1 x 392
        else:
            X_lime_sample = indicator_X0.to('cuda')
            weights_sample = X_weights.to('cuda')
        X_lime_list.append(X_lime_sample.unsqueeze(1))
        y_lime_list.append(logit0.unsqueeze(1))
        weights_list.append(weights_sample.unsqueeze(1))
    X_lime = torch.cat(X_lime_list, dim=1)
    y_lime = torch.cat(y_lime_list, dim=1)
    weights = torch.cat(weights_list, dim=1) 
    torch.set_grad_enabled(True)
    r_model = RidgeRegression(weights.size(2), alpha=1.0).to(device)
    optimizer = torch.optim.Adam(r_model.parameters(), lr=0.01)

    for epoch in range(nepochs):
        r_model.train()
        optimizer.zero_grad()
        
        loss = r_model.compute_loss(X_lime[0], y_lime[0], weights[0])
        loss.backward(retain_graph=True)
        optimizer.step()

    feature_importance = r_model.linear.weight.detach().cpu().numpy()
    feature_importance = torch.tensor(feature_importance).to(device)
    return feature_importance