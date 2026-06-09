

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from src import config

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, df, image_dir, training=True, transform = None):
         'Initialization'
         self.training = training
         self.image_dir = image_dir
         self.df = df
         #self.transform = transforms.ToTensor()
         self.transform = transform
         
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        
        # Select sample
        row = self.df.iloc[index]
        filename = row['filename']

        # Load data and get label
        img = Image.open(f"/{self.image_dir}/{filename}")

        # Converts image into a PyTorch tensor
        #X = self.transform(img)

        if self.transform:
            X = self.transform(img)
        else:
            X = transforms.ToTensor()(img)            

        if self.training:
            y = row['FaceOcclusion']
            y = np.float32(y)
            gender = row['gender']
            return X, y, gender, filename
        else:
            y = None
            gender = None
            return X, filename




if __name__ == "__main__":

    print()
    print("*****************************************************")
    print("dataset.py")
    print("*****************************************************")

    import pandas as pd
    
    df_train = pd.read_csv(config.DF_TRAIN, delimiter=',')
    df_test = pd.read_csv(config.DF_TEST, delimiter=',')

    # Remove nan values
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Data split
    df_val = df_train.iloc[:20000].reset_index(drop = True)
    df_train = df_train.iloc[20000:].reset_index(drop = True)

    # Make Dataloader
    training_set = Dataset(df_train, config.IMAGE_DIR)
    validation_set = Dataset(df_val, config.IMAGE_DIR)
    test_set = Dataset(df_test, config.IMAGE_DIR, training=False)

    print(f"TRAIN data contains {training_set.__len__()} samples.")    
    X, y, gender, filename = training_set.__getitem__(2)
    print(f"X = {type(X)}, {X.shape}")
    print(f"y = {y}")
    print(f"gender = {gender}")
    print(f"filename = {filename}\n")

    print(f"VALIDATION data contains {validation_set.__len__()} samples.")
    X, y, gender, filename = validation_set.__getitem__(2)
    print(f"X = {type(X)}, {X.shape}")
    print(f"y = {y}")
    print(f"gender = {gender}")
    print(f"filename = {filename}\n")  
    
    print(f"TEST data contains {test_set.__len__()} samples.")
    X, filename = test_set.__getitem__(2)
    print(f"X = {type(X)}, {X.shape}")
    print(f"filename = {filename}\n")

    print()
