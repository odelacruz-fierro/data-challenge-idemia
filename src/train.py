

from src import config
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd

def batch_gender_counts(gender_batch, epoch):
    """
    Analyse les batchs d'une époque donnée et affiche la distribution.
    """
    if epoch not in config.TARGET_EPOCHS:
        return None
    
    females_per_batch = int(torch.sum(gender_batch == 0.0))
    males_per_batch = int(torch.sum(gender_batch == 1.0))
    
    return females_per_batch, males_per_batch 

def train_one_epoch(model, dataloader, optimizer, criterion_occ, criterion_gender, gamma_gender, epoch):
    # Activate training mode
    model.train()

    # Loss trackers
    running_total = 0.0
    running_occ = 0.0
    running_gender = 0.0

    # Gender tracker
    gender_epoch_distribution = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training-Epoch {epoch}")

    for batch_idx, (X, y, gender, filename) in pbar:

        # --- Transfert to GPU ---
        X = X.to(config.DEVICE)
        y = y.to(config.DEVICE)
        y = y.view(-1, 1)
        gender = gender.to(config.DEVICE).float().view(-1, 1)

        # Reset gradients
        optimizer.zero_grad()

        # --- Forward pass ---
        pred_occ, pred_gender = model(X)

        # Compute loss
        loss_occ = criterion_occ(pred_occ, y)
        loss_gender = criterion_gender(pred_gender, gender)
        total_loss = loss_occ + gamma_gender*loss_gender

        if loss_occ.isnan():
            print(filename)
            print('label', y)
            print('y_pred', pred_occ)
            break

        # Update loss trackers  
        running_total += total_loss.item()
        running_occ += loss_occ.item()
        running_gender += loss_gender.item()

        pbar.set_postfix({'Loss': f"{total_loss.item():.4f}"})
        
        # Backpropagation
        total_loss.backward()
        # Update model weights
        optimizer.step()
        
        if epoch in config.TARGET_EPOCHS:
            f_batch_counts, m_batch_counts  = batch_gender_counts(gender, epoch)
            batch_counts = {
                'batch_idx':batch_idx,
                'f_counts': f_batch_counts,
                'm_counts': m_batch_counts
            }
            gender_epoch_distribution.append(batch_counts)

    # --- Write here the csv directly ---
    if epoch in config.TARGET_EPOCHS:
        print(f"\nSaving epoch {epoch} gender counts to =  {config.OUTPUT_DIR}/")
        gender_epoch_dist_df = pd.DataFrame(gender_epoch_distribution)
        gender_epoch_dist_df.to_csv(f"/{config.OUTPUT_DIR}/epoch_{epoch}_gender_dist_aug.csv", sep=',', index=False)

    train_metrics = {
        'loss_total': running_total / len(dataloader),
        'loss_occ': running_occ / len(dataloader),
        'loss_gender': running_gender / len(dataloader)
    }

    return train_metrics
        
        
        
if __name__ == "__main__":

    print()
    print("*****************************************************")
    print("train.py")
    print("*****************************************************")


