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
from torch.optim.lr_scheduler import CosineAnnealingLR


import torchvision
#from torchvision.models import mobilenet_v3_small


def main():
    print()
    print("*" * 60)
    print("#")
    print("#")    
    print("# Welcome to Data IADATA704 data challenge TRAIN and VALIDATION pipeline")
    print("# Powered By Idemia and Télécom Paris")
    print("#")
    print("#")
    print("#")
    print("*" * 60)

    print(f"Loading data from: {config.RAW_DATA_DIR}")    

    # ------------------------------------------------------------
    # Transformation pipelines
    # ------------------------------------------------------------
    train_transforms = T.Compose([
        # 1. Simulates off-center/cropped faces (Intervals 2 & 3 -> 0.40, 0.41)
        # Cropping is minimal (max 15%) to stay consistent with the dataset distribution
        T.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        
        # 2. Simulates head tilts (Interval 3 -> 0.46, Interval 4 -> 0.54)
        T.RandomRotation(degrees=15),
        
        # 3. Simulates heavy shadows and contrast changes (Interval 3 -> Trumpet, Interval 4 -> 0.71)
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        
        # 4. Standard geometric augmentations and blur
        T.RandomHorizontalFlip(p=0.5),
        T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.5)),
        
        # 5. Convert image to tensor
        T.ToTensor(),        
        
        # 6. Injection of mini black rectangles (simulating localized occlusion)
        T.RandomErasing(p=config.TRANSFORM_PROB, scale=(config.SCALE[0], config.SCALE[1]), ratio=(config.RATIO[0], config.RATIO[1]), value=0),
        
        # 7. Final ImageNet normalization for your ResNet-50 backbone
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------
    df_train = pd.read_csv(config.DF_TRAIN, delimiter=',')

    # --- Remove nan values ---
    df_train = df_train.dropna()
    
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
        tools.check_images_integrity(df_val, config.IMAGE_DIR)
    else:
        print("\nData check disabled ... /!\\ /!\\ /!\\")

    # ------------------------------------------------------------
    # Dataset and Dataloader
    # ------------------------------------------------------------
    training_set = dataset.Dataset(df_train, config.IMAGE_DIR, training =  True, transform = train_transforms)
    validation_set = dataset.Dataset(df_val, config.IMAGE_DIR, training = True, transform = val_transforms)  

    # --- Gender weights computation ---
    female_count = len(df_train[df_train['gender'] == 0.0])
    male_count = len(df_train[df_train['gender'] == 1.0])

    print(f"\n-> Females: {female_count}, Males: {male_count}")

    class_weights = {
        0.0: 1.0 / female_count,
        1.0: 1.0 / male_count
    }

    sample_weights = [class_weights[g] for g in df_train['gender']]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)

    # --- Create the Sampler ---
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True 
    )

    print("\nCreating dataloaders ...")
    training_generator = torch.utils.data.DataLoader(training_set,
                                                     batch_size = config.params_train['batch_size'],
                                                     sampler = sampler,
                                                     #shuffle = config.params_train['shuffle'],
                                                     num_workers = config.params_train['num_workers'])
    
    validation_generator = torch.utils.data.DataLoader(validation_set, **config.params_val)    

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    print("\nBuilding Model ...")

    if config.USE_CUDA:
        torch.backends.cudnn.benchmark = True    
    
    match config.MODEL:
        case "MultiTaskMobileNet":
            print(f"Model {config.MODEL} is being build")
            model = mtl.MultiTaskMobileNet().to(config.DEVICE)

        case "MultiTaskResNet50":
            print(f"Model {config.MODEL} is being build")  
            model = mtl.MultiTaskResNet50().to(config.DEVICE)
        case _:
            print("Not a valid model !!!")  # Default case


    # --- Display model information ---
    if config.MODEL_INFO:
        print(model)
        tools.count_parameters(model)

    # ------------------------------------------------------------
    # Training + Evaluation
    # ------------------------------------------------------------

    # --- Loss function ---
    criterion_occ = nn.MSELoss()
    criterion_gender = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,    # Garde ton LR de base (ex: 1e-4 ou 5e-5)
        weight_decay=1e-2           # Plus fort pour calmer les dents de scie
    )

    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config.NUM_EPOCHS,                   # Nombre total d'époques de ton run
        eta_min=1e-6                # Le LR minimum en fin de course (très proche de 0)
    )      

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

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # --- Epoch's data ---
        training_history.append({
            'epoch': epoch,
            'LR': current_lr,
            'gamma_gender':config.GAMMA_GENDER, 
            'idemia_val_score': idemia_val_score,
            'train_loss_total': train_metrics['loss_total'],
            'train_loss_occ': train_metrics['loss_occ'],
            'train_loss_gender': train_metrics['loss_gender'],
            'val_loss_total': val_metrics['loss_total'],
            'val_loss_occ': val_metrics['loss_occ'],
            'val_loss_gender': val_metrics['loss_gender'] 
        })        

        print(f"📉 Epoch {epoch} complete. Learning rate updated to: {current_lr:.6f}")
        
        if idemia_val_score < best_score:
            print(f"\n🔥🔥🔥New best score! ({best_score:.4f} --> {idemia_val_score:.4f}). Saving model...🔥🔥🔥")
            torch.save(model.state_dict(), f"/{config.OUTPUT_DIR}/best_model.pth")
            
            print(f"\nSaving validation_predictions.csv to =  {config.OUTPUT_DIR}/")
            val_results_df.to_csv(f"/{config.OUTPUT_DIR}/best_validation_predictions.csv", sep=',', index=False)

            best_score = idemia_val_score 

    # --- Export history ----
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(f"/{config.OUTPUT_DIR}/training_history.csv", sep=',', index=False)
    print(f"\nTraining complete! ✅\n\nSaving training_history.csv to {config.OUTPUT_DIR}/\n")

if __name__ == "__main__":
    main()