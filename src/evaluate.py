
import numpy as np
import torch
from tqdm import tqdm
from src import config
import pandas as pd


def error_fn(df):
    pred = df.loc[:, "pred"]
    ground_truth = df.loc[:, "target"]      # FaceOcclusion (label)
    weight = 1/30 + ground_truth

    return np.sum(((pred - ground_truth)**2) * weight, axis=0) / np.sum(weight, axis=0)

def metric_fn(female, male):
    err_male = error_fn(male)
    err_female = error_fn(female)
    return (err_male + err_female) / 2 + abs(err_male - err_female)


def validation_one_epoch(model, dataloader, criterion_occ, criterion_gender, gamma_gender, epoch):
    # Activate evaluation mode
    model.eval()

    # Loss trackers
    running_total = 0.0
    running_occ = 0.0
    running_gender = 0.0

    results_list = []    

    with torch.inference_mode():

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Validation-Epoch {epoch}")
        
        for batch_idx, (X, y, gender, filename) in pbar:

            # --- Transfert to GPU ---
            X = X.to(config.DEVICE)
            y = y.to(config.DEVICE)
            y = y.view(-1, 1)
            gender = gender.to(config.DEVICE).float().view(-1, 1)

            # --- Forward pass ---
            pred_occ, pred_gender = model(X)                # Occlusion
            #pred_gender_class = pred_gender.argmax(dim=1)   # Gender
            pred_gender_class = (pred_gender > 0).int()

            # --- Compute loss ---
            loss_occ = criterion_occ(pred_occ, y)
            loss_gender = criterion_gender(pred_gender, gender)
            total_loss = loss_occ + gamma_gender*loss_gender

            # --- Update loss trackers ---
            running_total += total_loss.item()
            running_occ += loss_occ.item()
            running_gender += loss_gender.item()

            # --- Move batch to cpu ---
            #pred_occ_cpu = pred_occ.detach().cpu()
            #y_cpu = y.detach().cpu()
            #gender_cpu = gender.detach().cpu()
            #pred_gender_class_cpu = pred_gender_class.detach().cpu()

            pred_occ_gpu = pred_occ.detach()
            
            # --- Gather outputs ---
            for i in range(len(X)):
                #results_list.append({
                #    'filename': filename[i],
                #    'pred': pred_occ_cpu[i].item(),
                #    'target': y_cpu[i].item(),
                #    'gender': gender_cpu[i].item(),
                #    'gender_pred': int(pred_gender_class_cpu[i][0].item())
                #})
            
                results_list.append({
                    'filename': filename[i],
                    'pred': pred_occ_gpu[i][0].item(),  # Ajout de [0] pour la dimension 2D
                    'target': y[i][0].item(),           # Idem
                    'gender': gender[i][0].item(),       # Idem
                    'gender_pred': pred_gender_class[i][0].item() # Idem
                })
        
        validation_metrics = {
            'loss_total': running_total / len(dataloader),
            'loss_occ': running_occ / len(dataloader),
            'loss_gender': running_gender / len(dataloader)
        }

        # --- Convert to dataframe ---
        results_df = pd.DataFrame(results_list)

        # --- Compute Idemia score ---
        results_male = results_df.loc[results_df["gender"] == 1.0]
        results_female = results_df.loc[results_df["gender"] == 0.0]
        idemia_score = metric_fn(results_female, results_male)        

        return idemia_score, results_df, validation_metrics


if __name__ == "__main__":

    print()
    print("*****************************************************")
    print("evaluate.py")
    print("*****************************************************")