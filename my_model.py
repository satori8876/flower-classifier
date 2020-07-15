from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import models

def build_model(arch, hidden_layers, out):
    if arch.lower() == 'alexnet':
        model = build_alexnet_model(hidden_layers, out)
    elif arch.lower().find('vgg') == 0:
        model = build_VGG_model(arch, hidden_layers, out)
    elif arch.lower().find('resnet') == 0:
        model = build_resnet_model(arch, hidden_layers, out)
    elif arch.lower().find('densenet') == 0:
        model = build_densenet_model(arch, hidden_layers, out)
       
    return model
        
def build_alexnet_model(hidden_layers, out):
    model = models.alexnet(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    inp = 9216
    model.classifier = build_layers(inp, out, hidden_layers)
    
    return model

def build_VGG_model(arch, hidden_layers, out):
    if arch.lower() == 'vgg11':
        model = models.vgg11(pretrained=True)
        
    if arch.lower() == 'vgg13':
        model = models.vgg13(pretrained=True)
        
    if arch.lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
        
    if arch.lower() == 'vgg19':
        model = models.vgg19(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    inp = 25088
    model.classifier = build_layers(inp, out, hidden_layers)
    
    return model

def build_resnet_model(arch, hidden_layers, out):
    if arch.lower() == 'resnet18':
        model = models.resnet18(pretrained=True)
        inp = 512
    if arch.lower() == 'resnet34':
        model = models.resnet34(pretrained=True)
        inp = 512
    if arch.lower() == 'resnet50':
        model = models.resnet50(pretrained=True)
        inp = 2048
    if arch.lower() == 'resnet101':
        model = models.resnet101(pretrained=True)
        inp = 2048
    if arch.lower() == 'resnet152':
        model = models.resnet152(pretrained=True)
        inp = 2048
        
    for param in model.parameters():
        param.requires_grad = False
       
    model.fc = build_layers(inp, out, hidden_layers)
    
    return model

def build_densenet_model(arch, hidden_layers, out):
    if arch.lower() == 'densenet121':
        model = models.densenet121(pretrained=True)
        inp = 1024
    if arch.lower() == 'densenet161':
        model = models.densenet161(pretrained=True)
        inp = 2208
    if arch.lower() == 'densenet169':
        model = models.densenet169(pretrained=True)
        inp = 1664
    if arch.lower() == 'densenet201':
        model = models.densenet201(pretrained=True)
        inp = 1920
        
    for param in model.parameters():
        param.requires_grad = False
       
    model.classifier = build_layers(inp, out, hidden_layers)
    
    return model

def build_layers(inp, out, hidden):
    
    # print("  New and improved build_layers : ")
    
    layer_dict = OrderedDict([
            ('fc1',  nn.Linear(inp, hidden[0])),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(p=0.2))])
    
    for i in range(len(hidden)-1):
        new_key = 'fc'+str(i+2) 
        layer_dict[new_key] = nn.Linear(hidden[i], hidden[i+1])
        layer_dict['relu']  = nn.ReLU()
        layer_dict['dropout'] = nn.Dropout(p=0.2)
        
    new_key = 'fc'+str(len(hidden)+1)
    layer_dict[new_key]  = nn.Linear(hidden[-1], out)
    layer_dict['output'] = nn.LogSoftmax(dim=1)
    
    return nn.Sequential(layer_dict)  

def build_checkpoint(arch, hidden_layers, out, model_state, optim_state, class_to_idx):
    
    inp = get_input_num(arch.lower())        
    
    checkpoint = {'input_size':    inp, 
                  'output_size':   out,
                  'hidden_layers': hidden_layers,  
                  'pretrained_model': arch, 
                  'state_dict':    model_state,
                  'optim_state':   optim_state,
                  'class_to_idx':  class_to_idx}
    
    return checkpoint

def get_input_num(a):
    
    if a.find('alexnet') == 0:
        inp = 9216
    elif a.find('vgg') == 0:
        inp = 25088
    elif a.find('resnet18') == 0 or a.find('resnet34') == 0:
        inp = 512
    elif a.find('resnet50') == 0 or a.find('resnet101') == 0 or a.find('resnet152') == 0:
        inp = 2048
    elif a.find('densenet121') == 0:
        inp = 1024
    elif a.find('densenet161') == 0:
        inp = 2208
    elif a.find('densenet169') == 0:
        inp = 1664
    elif a.find('densenet201') == 0:
        inp = 1920
        
    return inp    

def load_checkpoint(filepath):
 
    checkpoint = torch.load(filepath)
    model = build_model(checkpoint['pretrained_model'], 
                        checkpoint['hidden_layers'], 
                        checkpoint['output_size'] )
    model.load_state_dict(checkpoint['state_dict'])
    mapping = checkpoint['class_to_idx']
    
    return model, mapping

