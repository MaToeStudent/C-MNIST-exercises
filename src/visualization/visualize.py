import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

sys.path.insert(0,'src/models')

import logging
from pathlib import Path

import click
import torch
from dotenv import find_dotenv, load_dotenv
from torch import nn
from torch.utils.data import Dataset
##############################

import argparse
import sys

import torch
from torch import optim
from torch import nn
import numpy as np
import pickle
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.manifold import TSNE

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.output_activation = nn.LogSoftmax()
    
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x_last = self.dropout(self.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x_last), dim=1)

        return x_last


class ProcessCorruptMNIST(Dataset):
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



def extract_features():

    parser = argparse.ArgumentParser(description="Testing arguments")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data", type=str, default="data/processed/train.pth")
    parser.add_argument("--load_model_path", type=str, default="models/my_trained_model.pth")
    parser.add_argument("--save_features_path", type=str, default="data/interim")
    parser.add_argument("--result_file_path", type=str, default="src/visualization")
    args = parser.parse_args(sys.argv[1:])
    print(args)

    data = args.data
    load_model_path = args.load_model_path
    batch_size = args.batch_size
    save_features_path = args.save_features_path
    train_loss = args.result_file_path + "/train_loss.p"
    train_acc = args.result_file_path + "/train_acc.p"
    test_acc = args.result_file_path + "/test_acc.p"

    model = MyAwesomeModel()
    state_dict = torch.load(load_model_path)
    model.load_state_dict(state_dict)
    model.eval()
    test_set = torch.load(data).images
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    parameters = list(model.parameters())[-1].data.numpy()

    trainloss = pickle.load(open(train_loss,"rb"))
    trainacc = pickle.load(open(train_acc,"rb"))    
    accuracy = pickle.load(open(test_acc,"rb"))
    print('Test accuracy: ', accuracy,'%')
    plt.plot(range(1,len(trainloss)+1),trainloss,'-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train loss')
    plt.savefig('reports/figures/Train_loss.png')
    plt.close()
    plt.plot(range(1,len(trainacc)+1),trainacc,'-')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.title('Train Accuracy')
    plt.savefig('reports/figures/Train_acc.png')
    plt.close()
    
    plt.bar(x = range(1,len(parameters)+1),height=parameters)
    plt.xlabel('weights')
    plt.ylabel('Size of weight')
    plt.title('Weights in final layer')
    plt.savefig('reports/figures/weights_final_layer.png')
    plt.close()
    """
    traindata = torch.load(data).images
    traindata = traindata.view(traindata.shape[0],-1).data.numpy()
    print(traindata.shape)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(traindata)
    print(X_embedded)
    plt.plot(X_embedded[:,0],X_embedded[:,1],'*')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('t-SNE')
    plt.savefig('reports/figures/t_SNE.png')
    plt.close()
    """
    features = []
    with torch.no_grad():
        model.eval()
        for images in testloader: #, labels
            ps = torch.exp(model(images))
            features.append(ps.numpy().astype(np.float64))
            #print(ps.shape)
            #top_p, top_class = ps.topk(1, dim=1)
            #equals = top_class == labels.view(*top_class.shape)
            #accuracy = torch.mean(equals.type(torch.FloatTensor))
            #print(f'accuracy: {accuracy.item()*100}%')
    model.train()

    features = np.asarray(features)
    features = features.reshape(*features.shape[:-2], -1)
    print(features.shape)
    fileName = save_features_path + "/last_layer_features"
    fileObject = open(fileName, 'wb')

    pickle.dump(features, fileObject)
    fileObject.close()

    features_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(features)
    print(features_embedded.shape)


if __name__ == '__main__':
    extract_features()
