import pandas as pd

from src import config
from src import utils as tools
from src import dataset
from src import train
from src import evaluate
from src import model as mtl

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
    print("# By Oscar DE LA CRUZ")
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
    df_val = df_train.iloc[:20000].reset_index(drop = True)     # 20% Validation
    df_train = df_train.iloc[20000:].reset_index(drop = True)   # 80% Train
        
    # ------------------------------------------------------------
    # Data integrity check
    # ------------------------------------------------------------
    if config.DATA_CHECK:
        print("\nChecking TRAIN data integrity ...")
        tools.check_images_integrity(df_train, config.IMAGE_DIR)

        print("\nChecking VALIDATION data integrity ...")
        tools.check_imatraining_historyges_integrity(df_val, config.IMAGE_DIR)

        print("\nChecking TEST data integrity ...")
        tools.check_images_integrity(df_test, config.IMAGE_DIR)
    else:
        print("\nData check disabled ... /!\\ /!\\ /!\\")

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
    print("\nBuilding Model ...")

    if config.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    #model = torchvision.models.mobilenet_v3_small(num_classes=1)
    #model = model.to(config.DEVICE)

    model = mtl.MultiTaskMobileNet().to(config.DEVICE)

    # Display model information
    if config.MODEL_INFO:
        print(model)
        tools.count_parameters(model)

    # ------------------------------------------------------------
    # Training + Evaluation
    # ------------------------------------------------------------

    # --- Loss function ---
    criterion_occ = nn.MSELoss()
    criterion_gender = nn.CrossEntropyLoss()

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)

    # --- Best score ---
    best_score = float('inf')

    # --- Data history ---
    training_history = []

    for epoch in range(1, config.NUM_EPOCHS + 1):
        
        # --- Training ---
        print(f"\n[Epoch {epoch}/{config.NUM_EPOCHS}] Training started ...\n")

        train_metrics = train.train_one_epoch(
            model = model,
            dataloader = training_generator,
            optimizer = optimizer,
            criterion_occ = criterion_occ,
            criterion_gender = criterion_gender,
            gamma_gender = config.GAMMA_GENDER,
            epoch = epoch
        )

        # --- Validation ---
        print("\nEvaluation on validation split started ...\n")
        
        idemia_val_score, val_results_df, val_metrics = evaluate.validation_one_epoch(
            model = model,
            dataloader = validation_generator,
            criterion_occ = criterion_occ,
            criterion_gender = criterion_gender,
            gamma_gender = config.GAMMA_GENDER,
            epoch = epoch
        )

        # --- Epoch's data ---
        training_history.append({
            'epoch': epoch,
            'gamma_gender':config.GAMMA_GENDER, 
            'idemia_val_score': idemia_val_score,
            'train_loss_total': train_metrics['loss_total'],
            'train_loss_occ': train_metrics['loss_occ'],
            'train_loss_gender': train_metrics['loss_gender'],
            'val_loss_total': val_metrics['loss_total'],
            'val_loss_occ': val_metrics['loss_occ'],
            'val_loss_gender': val_metrics['loss_gender'] 
        })

        if idemia_val_score < best_score:
            print(f"\nNew best score! ({best_score:.4f} --> {idemia_val_score:.4f}). Saving model...")
            torch.save(model.state_dict(), f"/{config.OUOTPUT_DIR}/best_model.pth")
            
            print(f"\nSaving validation_predictions.csv to =  {config.OUOTPUT_DIR}/")
            val_results_df.to_csv(f"/{config.OUOTPUT_DIR}/best_validation_predictions.csv", sep=',', index=False)

            best_score = idemia_val_score

    # --- Export history ----
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(f"/{config.OUOTPUT_DIR}/training_history.csv", sep=',', index=False)
    print(f"\nTraining complete! ✅\n\nSaving training_history.csv to {config.OUOTPUT_DIR}/\n")

if __name__ == "__main__":
    main()