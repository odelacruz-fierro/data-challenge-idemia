
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


def evaluate_one_epoch(model, dataloader):

    # Activate evaluation mode
    model.eval()

    results_list = [] 
       
    if dataloader.dataset.training:

        with torch.inference_mode():  

            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation")
            
            for batch_idx, (X, y, gender, filename) in pbar:
                X = X.to(config.DEVICE)
                y = y.to(config.DEVICE)

                # --- Forward pass ---
                pred_occ, pred_gender = model(X)                # Occlusion
                pred_gender_class = pred_gender.argmax(dim=1)   # Gender
                
                # --- Gather outputs ---
                for i in range(len(X)):
                    results_list.append({'filename': filename[i],
                                        'pred': float(pred_occ[i]),
                                        'target': float(y[i]),
                                        'gender': float(gender[i])
                                        })    
        

        # --- Convert to dataframe ---
        results_df = pd.DataFrame(results_list)

        # --- Compute Idemia score ---
        results_male = results_df.loc[results_df["gender"] == 1.0]
        results_female = results_df.loc[results_df["gender"] == 0.0]

        idemia_score = metric_fn(results_female, results_male )
        results_df['idemia_score'] = idemia_score        

        return idemia_score, results_df


if __name__ == "__main__":

    print()
    print("*****************************************************")
    print("evaluate.py")
    print("*****************************************************")