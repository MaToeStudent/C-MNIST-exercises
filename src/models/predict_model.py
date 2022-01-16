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



def evaluate():

    parser = argparse.ArgumentParser(description="Testing arguments")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data", type=str, default="data/processed/test.pth")
    parser.add_argument("--load_model_path", type=str, default="models/my_trained_model.pth")
    args = parser.parse_args(sys.argv[1:])
    print(args)

    data = args.data
    load_model_path = args.load_model_path
    batch_size = args.batch_size

    model = MyAwesomeModel()
    state_dict = torch.load(load_model_path)
    model.load_state_dict(state_dict)
    test_set = torch.load(data)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    test_acc = []
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f'accuracy: {accuracy.item()*100}%')
            test_acc.append(accuracy.item()*100)
    model.train()
    test_acc = sum(test_acc)/len(test_acc)
    pickle.dump(test_acc, open("src/visualization/test_acc.p", "wb"))

if __name__ == '__main__':
    evaluate()