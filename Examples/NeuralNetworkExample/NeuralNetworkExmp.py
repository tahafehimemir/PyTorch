from random import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F #relu
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms



class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes): #(28x28 = 784)
        super(NeuralNetwork,self).__init__()
        self.layer1 = nn.Linear(in_features= input_size, out_features= 50)
        self.layer2 = nn.Linear(in_features= 50, out_features= num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

#lets check is it working with random created image size
#model = NeuralNetwork(784, 10)
#x = torch.randn(64,784) #64 mini batch size
#print(model(x).shape) #working torch.Size([64,10])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.01
batch_size = 32
num_epochs = 1

#data
training_dataset = datasets.MNIST(root = 'dataset/',
                                  train = True,
                                  transform=transforms.ToTensor(),
                                  download=True)
#when uploading data it is going to be numpy array so we transform tensor
training_dataloader = DataLoader(dataset = training_dataset,
                                 batch_size= batch_size,
                                 shuffle=True)

testing_dataset = datasets.MNIST(root = 'dataset/',
                                 train = False,
                                 transform=transforms.ToTensor(),
                                 download=True)
testing_dataloader = DataLoader(dataset= testing_dataset,
                                batch_size=batch_size,
                                shuffle=False)
#in test part don't use shuffle. This way will more accurate

#network
model = NeuralNetwork(input_size= input_size, num_classes = num_classes).to(device)

#loss-optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr= learning_rate)

#training
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(training_dataloader):
        #we want to see which batch index that we have so use enumerate
        #data = images, targets = correct digit for each label for each image
        data = data.to(device = device)
        targets = targets.to(device=device)

        #print(data.shape)
        #this gives us torch.Size([32,1,28,28]) but we want 784 flatten
        #get correct shape
        data = data.reshape(data.shape[0], -1)

        #forward pass
        scores = model(data)
        loss_f = loss_function(scores, targets)

        #backward
        optimizer.zero_grad()
        loss_f.backward()

        #gradient descent
        optimizer.step()

def checking_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training")
    else:
        print("Checking accuracy on testing")

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x , y in loader:
            x = x.to(device = device)
            y = y.to(device = device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            #we want to have the max of the second dimension so we can
            #either set 0 or 1 here. If we would look at the shapes of scores
            #this will be 32 images times 10 we want to know which one is the maximum of those
            #10 digits. If max value at the first one then it will be a digit zero
            _, predictions = scores.max(1) #second dimension index
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"{num_correct} /{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")
    model.train()


checking_accuracy(training_dataloader, model)
checking_accuracy(testing_dataloader, model)