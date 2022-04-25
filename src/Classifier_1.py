# Courtesy of https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

# Set dowload = False, training data from open datasets.
training_data = datasets.MNIST(
    root="./", # look for the MNIST dir in this current dir 
    train=True,
    download=False,
    transform=ToTensor(),
)

# Set download = False, test data from open datasets.
test_data = datasets.MNIST(
    root="./", # look for the MNIST dir in this current dir 
    train=False,
    download=False,
    transform=ToTensor(),
)

# sand 

batch_size = 100
input_size = 784  # 28x28 covert an =picture array to be a 1D tensor  
hiden_size = 100
num_classes = 10 # from 0 t0 9
num_epochs = 2 # so that the learnig doesn't take too long 
learning_rate = 0.001


# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'The device is {device}')
example = iter(train_dataloader)
samples , labels = example.next()





class NeuralNetwork(nn.Module):
    def __init__(self,input_size,hiden_size,num_classes):
        super(NeuralNetwork, self).__init__()
        
        # Creating layers 
        self.linear1 = nn.Linear(input_size,hiden_size)
        
        # Apply the activation function after the 1st layer 
        
        self.relu = nn.ReLU()
        
        # Linear layer 2 
        self.linear2 = nn.Linear(hiden_size,num_classes)
        

    def forward(self, x):
        out  = self.linear1(x) # gets the sample x
        out = self.relu(out) # get the previous 
        out = self.linear2(out)
        
        
        return out 

model = NeuralNetwork(input_size,hiden_size,num_classes)

# Loss Function and optimizer 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)


# training loop 

n_total_steps = len(train_dataloader)

# the traing loop 
for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_dataloader):
        # 100, 1, 28, 28 reshape to  100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # forward pass
        outputs = model(images)
        loss = criterion(outputs,labels)
        
        #backward pass
        optimizer.zero_grad()
        loss.backward() # Back propagation
        optimizer.step() # the update step, apdate the parameters for us 
        
        if (i+1) %100 == 100 :
            print(f'Epoch {epoch+1}/{num_epochs} , step {i+1}/{n_total_steps}, Loss = {loss.item():.4f}')
        
# Don't want to compute the gradient in all the steps 

with torch.no_grad():
    n_correct = 0 
    n_samsple = 0
    
    for j,(images,labels) in enumerate(test_dataloader):
        
        # reshape
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # determine the predition, using our trained model 
        
        outputs = model(images) # passing the test images 
        
        # get the actual preditions 
        
        #(value, index)
        _, predictions = torch.max(outputs,1) # along 1 
        
        n_samsple += labels.shape[0] # the number of samples in the current batch, i.e 100 per batch 
        
        n_correct += (predictions == labels).sum().item()
    
    # the accuracy 
    
    acc = 100.0 * n_correct/n_samsple
    
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
        
        
        
        