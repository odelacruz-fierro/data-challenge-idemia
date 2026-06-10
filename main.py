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
    print("# By Oscar DE LA CRUZ")
    print("#")
    print("#")
    print("*" * 60)

    print(f"Loading data from: {config.RAW_DATA_DIR}")
    

    # ------------------------------------------------------------
    # Transformation pipelines
    # ------------------------------------------------------------
    
    train_transforms = T.Compose([
        T.Resize((224, 224)), 
        
        # Apply Blur
        T.GaussianBlur(kernel_size=(config.KERNEL_SZ[0], config.KERNEL_SZ[1]), sigma=(config.SIGMA[0], config.SIGMA[1])),
        
        # Convert to Tensor (Required for Random Erasing)
        T.ToTensor(),        
        
        # scale => The Total Area (Size)
        # ratio => The Aspect Ratio (Shape) = Height / Width
        T.RandomErasing(p=config.TRANSFORM_PROB, scale=(config.SCALE[0], config.SCALE[1]), ratio=(config.RATIO[0], config.RATIO[1]), value=0),
        
        # From ImageNet
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    '''
    train_transforms = T.Compose([
        # 1. Simule les visages excentrés/coupés (Intervals 2 & 3 -> 0.40, 0.41)
        # On ne crop que très peu (max 15%) pour rester dans les clous du dataset
        T.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        
        # 2. Simule les inclinaisons de tête (Interval 3 -> 0.46, Interval 4 -> 0.54)
        T.RandomRotation(degrees=15),
        
        # 3. Simule les ombres massives et contrastes (Interval 3 -> Trumpet, Interval 4 -> 0.71)
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
        
        # 4. Augmentations géométriques et flou standards
        T.RandomHorizontalFlip(p=0.5),
        T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 1.5)),
        
        # 5. Passage en tenseur
        T.ToTensor(),        
        
        # 6. Injection des mini-rectangles noirs
        T.RandomErasing(p=config.TRANSFORM_PROB, scale=(config.SCALE[0], config.SCALE[1]), ratio=(config.RATIO[0], config.RATIO[1]), value=0),
        
        # 7. Normalisation ImageNet finale pour ton ResNet50
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    '''

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

    # 4. Create the Sampler
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
    # optimizer = torch.optim.Adam(model.parameters(), lr = config.LEARNING_RATE)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = config.LEARNING_RATE,
        weight_decay = 1e-4  # Standard value for regularization
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
            print(f"\n🔥🔥🔥New best score! ({best_score:.4f} --> {idemia_val_score:.4f}). Saving model...🔥🔥🔥")
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