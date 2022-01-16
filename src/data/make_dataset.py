# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import wget
import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen
import subprocess
import numpy as np
import glob
import torch
from torch.utils.data import Dataset

urls = ['https://github.com/SkafteNicki/dtu_mlops/blob/main/data/corruptmnist/test.npz', 
            'https://github.com/SkafteNicki/dtu_mlops/blob/main/data/corruptmnist/train_0.npz',
            'https://github.com/SkafteNicki/dtu_mlops/blob/main/data/corruptmnist/train_1.npz',
            'https://github.com/SkafteNicki/dtu_mlops/blob/main/data/corruptmnist/train_2.npz']

filenames = ['/test.npz','/train_0.npz','/train_1.npz','/train_2.npz']

dict_url_filename = dict(zip(filenames, urls))

class DownloadCorruptMNIST():
    def __init__(self, dict_url_filename, input_filepath):
        self.path = input_filepath
        self.dict_url_filename = dict_url_filename
    def download(self):
        for filename, url in self.dict_url_filename.items():
            if filename not in os.listdir(self.path):
                self.filepath = wget.download(url, out=self.path)
                print(self.filepath, 'Download finished!')
            else:
                print(filename, ' already in raw data folder')

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



    

@click.command()
@click.argument('input_filepath', type=click.Path()) #,default = '/data/raw')
@click.argument('output_filepath', type=click.Path())#, default = '/data/processed')
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
        
    #data = DownloadCorruptMNIST(dict_url_filename, input_filepath)
    #data.download()

    processed_train_data = ProcessCorruptMNIST(input_filepath, output_filepath)
    processed_train_data.MergeTrain()
    processed_test_data = ProcessCorruptMNIST(input_filepath, output_filepath)
    processed_test_data.CreateTest()
    torch.save(processed_train_data, processed_train_data.output_path + '/train.pth')
    torch.save(processed_train_data, processed_test_data.output_path + '/test.pth')

    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())


    main()
