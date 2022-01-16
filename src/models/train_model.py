import argparse
import sys
import torch
from torch import optim
from torch import nn
from model import MyAwesomeModel
import numpy as np
import pickle
from torch.utils.data import Dataset

class ProcessCorruptMNIST(Dataset):
    """
    This class is not used, but had to be imported to avoid the script crashing (when loading 
    the stored data/train.pth an error arises if this class can not be found)
    """
    def __init__(self, input_filepath, output_filepath):
        self.input_path = input_filepath
        self.output_path = output_filepath
    def MergeTrain(self):
        self.trainfiles_path = self.input_path + '/train'
        self.train_files = glob.glob(self.trainfiles_path + "*")
        train_data = []
        train_targets = []
        for file in self.train_files:
            file_load = np.load(file)
            train_data.append(file_load['images'])
            train_targets.append(file_load['labels'])
            
        self.images = torch.tensor(np.concatenate(train_data),dtype=torch.float)
        self.labels = torch.tensor(np.concatenate(train_targets),dtype=torch.float)
    
    def CreateTest(self):
        self.testfile_path = self.input_path + '/test.npz'
        test_data = np.load(self.testfile_path)
        self.images = torch.tensor(test_data['images'],dtype=torch.float)
        self.labels = torch.tensor(test_data['labels'],dtype=torch.float)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        label = self.labels[idx]
        img = self.images[idx]
        return img,label



def train():

    """Function to train a simple network on the Corrupt MNIST dataset
    Requirements:
        - Training data has to be available from 'data/raw/' created with 'make data' command from root-folder before this script can be executed

    Parameters:
        (OPTIONAL)
        --lr: learning rate (Default=0.003)
        --batch_size: batch size (Default=64)
        --weight_decay: Weight decay for optimizer (Default=1e-3)
        --data: string specifying path to data (Default="data/processed/train.pth")
        --save_model_path: string specifying where to store the trained model (Default="models/my_trained_model.pth")
        --optimizer: string specifying which optimizer to use (Default="Adam")
        --n_epochs: number of epochs/training rounds (Default=15)

    RETURNS:
        - models/my_trained_model.pth
        - src/visualization/train_loss.p
        - src/visualization/train_acc.p
    """
    parser = argparse.ArgumentParser(description="Training arguments")
    # Adding arguments
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--data", type=str, default="data/processed/train.pth")
    parser.add_argument("--save_model_path", type=str, default="models/my_trained_model.pth")
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--n_epochs', default=15)
    args = parser.parse_args(sys.argv[1:])
    print(args)

    model = MyAwesomeModel()
    model.train()        
    
    # Defining args
    data = args.data
    epochs = args.n_epochs
    save_model_path = args.save_model_path
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    lr = args.lr
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else: 
        optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Using NLL as criterion. Is converted with torch.exp() when model returns output. Equal to CrossEntropyLoss
    criterion = nn.NLLLoss()
    
    # Loading data and defining trainloader
    train_set = torch.load(data)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Creating list to store results
    train_loss = []
    train_acc = []
    
    
    for e in range(epochs):
        running_loss = 0
        running_acc = 0
        for images, labels in trainloader:
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Converting labels
            labels = labels.long()

            # Converting with torch.exp() to get CrossEntropyLoss
            output = torch.exp(model(images))
            
            # Computing loss
            loss = criterion(output, labels)
            
            # Backpropagation
            loss.backward()
            
            # New step for the optimizer
            optimizer.step()
            
            # Adding loss to running loss 
            running_loss += loss.item()
            
            # Calculate predictions
            y_pred = (nn.Softmax(dim=1)(output)).argmax(dim=1)
            running_acc += torch.sum(y_pred == labels).item() / labels.shape[0]
        
        # Append results to train loss and accuracy
        train_loss.append(running_loss)
        train_acc.append(running_acc / len(trainloader) * 100)
        print("Train loss: ", running_loss, "\ttrain accuracy: ", running_acc / len(trainloader) * 100, "%")


    # Saving outputs/results
    torch.save(model.state_dict(), save_model_path)
    pickle.dump(train_loss, open("src/visualization/train_loss.p", "wb"))
    pickle.dump(train_acc, open("src/visualization/train_acc.p", "wb"))


    

if __name__ == "__main__":
    train()
