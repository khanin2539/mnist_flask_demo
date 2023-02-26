import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

from flask import Flask 
from flask import request 
import io
import pickle
import sys
import numpy as np
import pandas as pd
import random
from PIL import Image
print(torch.__version__)

from torchvision import datasets, transforms

# Define a transform to normalize the data

# plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

from torch import nn

# Layer details for the neural network

# model class definition and architectures
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                        nn.ReLU(),
                        nn.Linear(hidden_sizes[1], output_size),
                        nn.LogSoftmax(dim=1))
# training function
def retrain(lr, num_epoch, bs):
  # transform / augment images 
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                                ])
    # train and validation data split
    trainset = datasets.MNIST('train/', download=True, train=True, transform=transform)
    valset = datasets.MNIST('val/', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=bs, shuffle=True)
    images, labels = next(iter(trainloader))

   
    # check input and label shapes 
   
    print(type(images))
    print(images.shape)
    print(labels.shape)
    # set model training loss functions and hyperpatameters
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    time0 = time()
    epochs = num_epoch
    # train
    for e in range(epochs):
        running_loss = 0
        num_correct = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            # Training pass
            optimizer.zero_grad()
            
            output = model(images)
            loss = criterion(output, labels)
            
            #This is where the model learns by backpropagating
            loss.backward()
            
            #And optimizes its weights here
            optimizer.step()
            
            running_loss += loss.item()
            num_correct += int((torch.argmax(output, axis=1) == labels).sum())
        # acc = 100 * num_correct / (len(trainloader) * bs)
        
        # else:
        acc = round(100 * num_correct / (len(trainloader) * bs), 2)
        print("Epoch {} - Training loss: {} - Training Accuracy: {}".format(e, running_loss/len(trainloader),acc))
        correct_count, all_count = 0, 0
        
        # validation step
        for images,labels in valloader:
            for i in range(len(labels)):
                img = images[i].view(1, 784)
                # Turn off gradients to speed up this part
                with torch.no_grad():
                    logps = model(img)

                # Output of the network are log-probabilities, need to take exponential for probabilities
                ps = torch.exp(logps)
                probab = list(ps.numpy()[0])
                pred_label = probab.index(max(probab))
                true_label = labels.numpy()[i]
                if(true_label == pred_label):
                    correct_count += 1
                all_count += 1

        print("Number Of Images Tested: ", all_count)
        print("Test Accuracy: ", (correct_count/all_count))
    test_acc = correct_count/all_count*100
    print("\nTraining Time (in minutes) =",(time()-time0)/60)


    # Save the model's state dictionary to a file
    torch.save(model.state_dict(), 'mnist_try.pth')
    return acc, test_acc


#validation
def validate():
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                                ])
    valset = datasets.MNIST('val/', download=True, train=False, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
    model.load_state_dict(torch.load("mnist.pth"))
    correct_count, all_count = 0, 0
    for images,labels in valloader:
      for i in range(len(labels)):
        img = images[i].view(1, 784)
        # Turn off gradients to speed up this part
        with torch.no_grad():
            logps = model(img)

        # Output of the network are log-probabilities, need to take exponential for probabilities
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))

# validate()
# test 1 image

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

  # test function with sample images
def predict(ts):
  # load the trained model
    model.load_state_dict(torch.load("mnist_try.pth"))
    
    ts = ts.view(1, 784)
    with torch.no_grad():
        logps = model(ts)
    # Output of the network are log-probabilities, need to take exponential for probabilities
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    # print("Predicted Digit =", probab.index(max(probab)))
    return  probab.index(max(probab))
# print(predict("test/10.png"))
