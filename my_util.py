
import numpy as np
from PIL import Image
import torch
from torchvision import datasets, transforms

indices = {}

def get_data_loaders(data_dir, print_it=False):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir  = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(0.1),
                                          transforms.RandomGrayscale(0.1),
                                          transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_transform  = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir,  transform=test_transform)
    test_data  = datasets.ImageFolder(test_dir,  transform=test_transform)

    if not data_dir in indices :
        print('Saving class_to_idx')
        indices[data_dir] = train_data.class_to_idx
        
    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_data,  batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,  batch_size=32, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_data,   batch_size=32, shuffle=False)

    if print_it:
        print("Training data: {} batches".format(len(train_loader)))
        print("Validation data: {} batches".format(len(valid_loader)))
        print("Testing data: {} batches".format(len(test_loader)))
    return train_loader, valid_loader, test_loader

def get_class_to_idx(data_dir):
    print('Retrieving class_to_idx')
    return indices[data_dir]

def get_output_length(data_dir):
    return len(indices.get(data_dir, 102))  

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    image.load()
    
    w = image.width
    h = image.height
    
    if w > h:
        width = 256 * w/h
        height = 256
    else:
        width = 256
        height = 256 * h/w
    
    image.thumbnail((width, height))
    
    left  = (image.width - 224) / 2
    upper = (image.height - 224) / 2
    right = left + 224
    lower = upper + 224
    image = image.crop((left, upper, right, lower))
        
    np_img = np.asarray(image)
    np_img = np_img / 255    
    means = [0.485, 0.456, 0.406]
    stds  = [0.229, 0.224, 0.225]
    norm_image = (np_img - means) / stds
     
    return norm_image.transpose((2, 0, 1))   


# get_data_loaders("flowers", print_it = True)

#img_path = './flowers/test/102/image_08023.jpg'
#my_image = Image.open(img_path)
#new_image = process_image(my_image)
#print(new_image[:3, :3, :3])