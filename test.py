import os

import pandas as pd

from src import config
from src import utils as tools
from src import dataset
from src import train
from src import evaluate
from src import model as mtl
from torch.utils.data import WeightedRandomSampler
import torchvision.transforms as T


import torch
import torch.nn as nn
from tqdm import tqdm


import torchvision
#from torchvision.models import mobilenet_v3_small


def main():
    print()
    print("*" * 60)
    print("#")
    print("#")    
    print("# Welcome to Data IADATA704 data challenge TEST pipeline")
    print("# Powered By Idemia and Télécom Paris")
    print("#")
    print("#")
    print("#")
    print("*" * 60)

    print(f"Loading data from: {config.RAW_DATA_DIR}")
    
    val_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------
    df_test = pd.read_csv(config.DF_TEST, delimiter=',')
        
    # ------------------------------------------------------------
    # Data integrity check
    # ------------------------------------------------------------
    if config.DATA_CHECK:
        print("\nChecking TEST data integrity ...")
        tools.check_images_integrity(df_test, config.IMAGE_DIR)
    else:
        print("\nData check disabled ... /!\\ /!\\ /!\\\n")

    # ------------------------------------------------------------
    # Dataset and Dataloader
    # ------------------------------------------------------------
    test_set = dataset.Dataset(df_test, config.IMAGE_DIR, training=False, transform=val_transforms)
    test_generator = torch.utils.data.DataLoader(test_set, **config.params_val)

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------
    print("#=" * 30)
    print("#")
    print("# Model selection")
    print("#")
    print("# Please select the model number you would like to test ...")
    print("#")
    print("#=" * 30)

    EXPERIMENTS = [folder.name for folder in os.scandir(f"/{config.OUTPUT_DIR}") if folder.is_dir()]
    EXPERIMENTS.sort()    

    print('\n')
    for index, EXP in enumerate(EXPERIMENTS):
        print(f"Model no.{index + 1} ==> {EXP}")
    print('\n')

    selected_model = input("Type the number of the model you want to test => ")
    selected_model = int(selected_model)
    print(f"Selected model ==> {EXPERIMENTS[selected_model - 1]}")

    MODEL_PATH = f"/{config.OUTPUT_DIR}/{EXPERIMENTS[selected_model - 1]}/best_model.pth"

    if os.path.exists(MODEL_PATH):
        print(f"Model FOUND!")
    else:
        print(f"Model NOT FOUND")
        exit()

    
    # --- Build model ---
    print("\nBuilding model and loading weights...")    

    match config.MODEL:
        case "MultiTaskMobileNet":
            print(f"Model {config.MODEL} is being build")
            model = mtl.MultiTaskMobileNet().to(config.DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH))

        case "MultiTaskResNet50":
            print(f"Model {config.MODEL} is being build")
            model = mtl.MultiTaskResNet50().to(config.DEVICE)
            model.load_state_dict(torch.load(MODEL_PATH))

        case _:
            print("Not a valid model !!!")  # Default case
    
    model.load_state_dict(torch.load(MODEL_PATH))

    model.eval()
    results_list = []

    with torch.inference_mode():
        for batch_idx, (X, filename) in tqdm(enumerate(test_generator), total=len(test_generator)):

            X = X.to(config.DEVICE)
            pred_occ, pred_gender = model(X)
        
            for i in range(len(X)):
                results_list.append({'filename': filename[i],
                                    'FaceOcclusion': float(pred_occ[i]),
                                    })
                
    results_df = pd.DataFrame(results_list)

    print(results_df.head(5))

    # --- Export predictions ---
    results_df['gender'] = 'x'
    results_df.to_csv(f"/{config.OUTPUT_DIR}/{EXPERIMENTS[selected_model - 1]}/test_predictions.csv", sep=',', index=False)
 

if __name__ == "__main__":
    main()