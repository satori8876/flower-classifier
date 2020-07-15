import argparse

import json
import math
import torch
import numpy as np
from PIL import Image

import my_model
import my_util 


parser = argparse.ArgumentParser(description='Get arguments for predicting image class.')
parser.add_argument('path_to_img', action='store')
parser.add_argument('checkpoint', action='store')
parser.add_argument('--top_k',   action='store', dest='top_k', type=int, default=3)
parser.add_argument('--category_names', action='store', dest='cat_names', default='cat_to_name.json')
parser.add_argument('--gpu',   action='store_true', default=False)

args = parser.parse_args()


def predict(image_path, checkpoint, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    img = Image.open(image_path)
    proc_image = my_util.process_image(img)
    tensor_img = torch.from_numpy(proc_image).float()
    tensor_img.unsqueeze_(dim=0)
    
    model, mapping = my_model.load_checkpoint(checkpoint)
    
    if torch.cuda.is_available() and args.gpu:
        print("  ===>>  move to GPU")
        model.cuda()
        tensor_img = tensor_img.cuda()

    model.eval()
    with torch.no_grad():
        output = model(tensor_img)
        probs  = torch.exp(output)
    
    probs, classes = probs.topk(topk, dim=1)

    if torch.cuda.is_available() and args.gpu:
        print("  ===>>  move to back to CPU")
        probs   = probs.cpu()
        classes = classes.cpu()
        
    probs_arr = probs.numpy()[0]
    class_arr = classes.numpy()[0]

    with open(args.cat_names, 'r') as f:
        name_dict = json.load(f)

    name_arr = []
    for clas in class_arr:
        for key, value in mapping.items():
            if value == clas:
                name_arr.append(key)

    class_names = [ name_dict[name] for name in name_arr]
    class_probs = [ "{:0.5f}%".format(x*100) for x in probs_arr]
    
    for i in  range(topk):
        print('{}: {}'.format(class_names[i], class_probs[i]))
        
#img_path = './flowers/test/102/image_08023.jpg'  
predict(args.path_to_img, args.checkpoint, args.top_k)

