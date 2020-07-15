import argparse

import time
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import  models

import my_model
import my_util

parser = argparse.ArgumentParser(description='Get arguments for training network.')

# most defaults are based on my previous jupyter notebook exercise
parser.add_argument('data_dir',   action='store')
parser.add_argument('--save_dir', action='store', dest='save_dir', default='.')
parser.add_argument('--arch',     action='store', dest='arch', default='resnet50')
parser.add_argument('--hidden_units',  nargs='+', dest='hidden_layers', type=int, default=[256,128])
parser.add_argument('--learning_rate', action='store', dest='lrate', type=float, default=0.01)
parser.add_argument('--test_every',    action='store', dest='test_every', type=int, default=67)
parser.add_argument('--epochs',   action='store',      dest='epochs', type=int, default=4)
parser.add_argument('--gpu',      action='store_true', default=False)

args = parser.parse_args()

train_loader, valid_loader, test_loader  = my_util.get_data_loaders(args.data_dir, print_it=True)

model = my_model.build_model(args.arch, args.hidden_layers, my_util.get_output_length(args.data_dir))
criterion = nn.NLLLoss()

classifier = model.fc if args.arch.lower().find('resnet') == 0 else model.classifier
optimizer  = optim.Adam(classifier.parameters(), lr=args.lrate)

device_is_cuda = torch.cuda.is_available() and args.gpu

if device_is_cuda:
    print("  ==> moving to GPU")
    model.cuda()

# python train.py flowers --learning_rate 0.003 

epochs     = args.epochs
test_every = args.test_every
# print(args)
train_loss = 0

t0 = time.time()

for e in range(epochs):
    for counter, (inputs, labels) in enumerate(train_loader):
        
        inputs = Variable(inputs)
        labels = Variable(labels)

        if device_is_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
            
        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()    

        optimizer.step()

        train_loss += loss.item()

        # validation
        if counter and counter % test_every == 0:
            # evaluation mode -- turn off backprop 
            model.eval()  
            test_loss = 0
            accuracy  = 0

            for images, labels in valid_loader:
                
                if device_is_cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                    
                output = model(images)
                loss   = criterion(output, labels)

                test_loss += loss.item() 

                # turn all output values positive 
                ps = torch.exp(output)
                # highest probability values for each image and their label indices
                top_ps, top_class = ps.topk(1, dim=1)

                # compare highest probability indices with expected indices 
                matches = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(matches.type(torch.FloatTensor))

            t1 = time.time()    
            mins = math.floor((t1-t0)/60) 
            secs = t1-t0-mins*60   
            print(f"  Time passed: {mins} mins {secs:.2f} secs ")
            print("  Train loss: {}".format(train_loss/test_every)) 
            print("  Test loss:  {}".format(test_loss/len(valid_loader))) 
            print("  Accuracy:   {} \n".format(accuracy/len(valid_loader)))
    
            train_loss = 0     
    
            # training mode -- turn backprop back on
            model.train()  
         

t1 = time.time()   
mins = math.floor((t1-t0)/60)  
secs = t1-t0-mins*60  
print(f"Total time:  {mins} mins {secs:.2f} secs ")
print("Average time per batch: {0:.3f} seconds".format((t1-t0)/(counter+1))) 

# Do final testing on the test set  
model.eval()
test_loss = 0
accuracy = 0
 
t0 = time.time()    

for images, labels in test_loader:
    
    if device_is_cuda:
        images = images.cuda()
        labels = labels.cuda()
    
    output = model(images)
    loss = criterion(output, labels)

    test_loss += loss.item()

    ps = torch.exp(output)
    top_ps, top_class = ps.topk(1, dim=1)

    matches = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(matches.type(torch.FloatTensor))
        
t1 = time.time()    
mins = math.floor((t1-t0)/60)  
secs = t1-t0-mins*60   
print(f"  Test time: {mins} mins {secs:.2f} secs ")
print("  Test loss:  {}".format(test_loss/len(test_loader)))
print("  Accuracy:   {} \n".format(accuracy/len(test_loader)))

checkpoint = my_model.build_checkpoint(args.arch, args.hidden_layers, 
                              my_util.get_output_length(args.data_dir),         
                              model.state_dict(), 
                              optimizer.state_dict, 
                              my_util.get_class_to_idx(args.data_dir))

torch.save(checkpoint, args.save_dir + '/my_checkpoint2.pth')
        
print(f"Saved checkpoint")