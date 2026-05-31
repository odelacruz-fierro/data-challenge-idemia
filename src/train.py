

from src import config
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, loss_fn, epoch):

    # Activate training mode
    model.train()

    current_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")

    for batch_idx, (X, y, gender, filename) in pbar:
        # Transfert to GPU
        X = X.to(config.DEVICE)
        y = y.to(config.DEVICE)
        y = y.view(-1, 1)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(X)

        # Compute loss
        loss = loss_fn(y_pred, y)

        if loss.isnan():
            print(filename)
            print('label', y)
            print('y_pred', y_pred)
            break
        
        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        current_loss += loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

    avg_loss = current_loss / len(dataloader)

    return avg_loss
        
        
        



