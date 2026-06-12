

from src import config
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, criterion_occ, criterion_gender, gamma_gender, epoch):
    # Activate training mode
    model.train()

    # Loss trackers
    running_total = 0.0
    running_occ = 0.0
    running_gender = 0.0

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


