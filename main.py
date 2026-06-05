import pandas as pd
from src import config
from src import utils as tools
from src import dataset
from src import train
from src import evaluate

import torch
import torch.nn as nn

import torchvision
#from torchvision.models import mobilenet_v3_small


def main():
    print()
    print("*" * 60)
    print("#")
    print("#")    
    print("# Welcome to Data IADATA704 data challenge pipeline")
    print("# Powered By Idemia and Télécom Paris")
    print("#")
    print("# BY Oscar DE LA CRUZ")
    print("#")
    print("#")
    print("*" * 60)

    print(f"Loading data from: {config.RAW_DATA_DIR}")
    
    # ------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------
    df_train = pd.read_csv(config.DF_TRAIN, delimiter=',')
    df_test = pd.read_csv(config.DF_TEST, delimiter=',')

    # --- Remove nan values ---
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    # --- Data split ---
    df_val = df_train.iloc[:20000].reset_index(drop = True)     # Validation
    df_train = df_train.iloc[20000:].reset_index(drop = True)   # Train
        
    # ------------------------------------------------------------
    # Data integrity check
    # ------------------------------------------------------------
    if config.DATA_CHECK:
        print("\nChecking TRAIN data integrity ...")
        tools.check_images_integrity(df_train, config.IMAGE_DIR)

        print("\nChecking VALIDATION data integrity ...")
        tools.check_images_integrity(df_val, config.IMAGE_DIR)

        print("\nChecking TEST data integrity ...")
        tools.check_images_integrity(df_test, config.IMAGE_DIR)
    else:
        print("\nDATA CHECK Disabled ... /!\\")

    # ------------------------------------------------------------
    # Dataset and Dataloader
    # ------------------------------------------------------------
    training_set = dataset.Dataset(df_train, config.IMAGE_DIR)
    validation_set = dataset.Dataset(df_val, config.IMAGE_DIR)
    test_set = dataset.Dataset(df_test, config.IMAGE_DIR, training=False)

    print("\nCreating dataloaders ...")
    training_generator = torch.utils.data.DataLoader(training_set, **config.params_train)
    validation_generator = torch.utils.data.DataLoader(validation_set, **config.params_val)
    test_generator = torch.utils.data.DataLoader(test_set, **config.params_val)

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    print("\nBuilding MODEL ...")

    if config.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    model = torchvision.models.mobilenet_v3_small(num_classes=1)
    model = model.to(config.DEVICE)
    
    # Display model information
    #print(model)
    #tools.count_parameters(model)

    # Loss function
    loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ------------------------------------------------------------
    # Training + Evaluation
    # ------------------------------------------------------------
    for epoch in range(1, config.NUM_EPOCHS + 1):
        
        # Training
        print("\nTRAINING started ...")
        avg_train_loss = train.train_one_epoch(
            model = model,
            dataloader = training_generator,
            optimizer = optimizer,
            loss_fn = loss_fn,
            epoch = epoch
        )

        # Evaluation
        print("\nEVALUATION started ...")
        idemia_val_score, val_results_df = evaluate.evaluate_one_epoch(
            model = model,
            dataloader = validation_generator
        )

        print(f"End of epoch {epoch} | Average loss : {avg_train_loss}")
        print(f"Train loss     : {avg_train_loss}")
        print(f"Idemia score   : {idemia_val_score}")
    

if __name__ == "__main__":
    main()