# Courtesy of https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# Courtesy of https://colab.research.google.com/drive/1jrKpcF6AVCh1M6_2aW9j-QpWnzOZh_mh?usp=sharing#scrollTo=6APZIZ6hyJYk

import torch
from torch import nn, sigmoid, softmax
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data

def main():
    # Download training data from open datasets.
    
    global training_data
    training_data = datasets.MNIST(
        root="./",
        train=True,
        download=False,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    global test_data
    test_data = datasets.MNIST(
        root="./",
        train=False,
        download=False,
        transform=ToTensor(),
    )


  

    # Get cpu or gpu device for training.
    
    global device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
  
    
    
    model = NeuralNetwork().to(device)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    Run(model,loss_fn,optimizer)
    
    print('Pytorch Output...')
    print('Done!')
        
    fn = input("Please enter a filepath:\n>")
        
    while fn != 'exit':
            
        img = Image.open(fn)
        # Define a transform to convert PIL 
        # image to a Torch tensor
        to_tensor = transforms.ToTensor()
        tensor = to_tensor(img).unsqueeze(0)
        pred = model(tensor)
        print(f'Classifier : {pred.argmax()}')
        fn = input("Please enter a filepath:\n>")
            
        
def Run(model,cost,opt):
    """
    Trains the models, and also auto detects when to stop i.e the number of epochs 
    """
    
    train_loss = []
    validation_acc = []
    best_model = None
    best_acc = None
    best_epoch = None
    max_epoch = 10000
    no_improvement = 5
    batch_size = 512

    
    for n_epoch in range(max_epoch):
        model.train()
        loader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=1)
        epoch_loss = []
        # traing loop
        for X_batch, y_batch in loader:
            opt.zero_grad()
            logits = model(X_batch)
            loss = cost(logits, y_batch)
            loss.backward()
            opt.step()        
            epoch_loss.append(loss.detach())
        train_loss.append(torch.tensor(epoch_loss).mean())
        model.eval()
        
        # Validating the model 
        loader = data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)
        X, y = next(iter(loader))
        logits = model(X)
        acc = compute_acc(logits, y).detach()
        validation_acc.append(acc)

        if best_acc is None or acc > best_acc:
            print("New best epoch ", n_epoch, "acc", acc)
            best_acc = acc
            best_model = model.state_dict()
            best_epoch = n_epoch
            
        if best_epoch + no_improvement <= n_epoch:
            print("No improvement for", no_improvement, "epochs")
            break
    model.load_state_dict(best_model)
    
    
    plt.title("Accuracy on the validation set")
    plt.plot(validation_acc)
    #plt.show()

    



def compute_acc(logits, expected):
    """
   
    In each row of logits,select the column number with the largest value, 
    compare it with the expected value, and average the values so obtained (0 or 1) 
    over the examples to obtain the classification accuracy.
    
    """
    pred = logits.argmax(dim=1)
    return (pred == expected).type(torch.float).mean()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten() # covert the 2D array to a 1D 784 size
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # input-to-hiden layer 1
            nn.Sigmoid(),
            
            nn.Linear(512,128),
            nn.Sigmoid(),
            
            nn.Linear(128,10),
            #nn.Soft(dim =1 )
         
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
if __name__ == '__main__':
    main()
