
from PIL import Image
from tqdm import tqdm
from src import config
from prettytable import PrettyTable

def check_images_integrity(df,image_dir):

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            filename = df.loc[idx, 'filename']
            #print(filename)
            img2display = Image.open(f"/{image_dir}/{filename}")
        except ValueError as e:
            print(idx, e)

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_trainable_params = 0
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        if not parameter.requires_grad:
            continue
        table.add_row([name, params])
        total_trainable_params += params
    print(table)
    print(f"Total Trainable Params: {total_trainable_params}")
    print(f"Total Params: {total_params}")



if __name__ == "__main__":

    print()
    print("*****************************************************")
    print("utils.py")
    print("*****************************************************")

    import pandas as pd
    
    df_train = pd.read_csv(config.DF_TRAIN, delimiter=',')
    df_test = pd.read_csv(config.DF_TEST, delimiter=',')

    # Remove nan values
    df_train = df_train.dropna()
    df_test = df_test.dropna()

    # Data split
    df_val = df_train.iloc[:20000].reset_index(drop = True)
    df_train = df_train.iloc[20000:].reset_index(drop = True)

    # -------------------------------------------------
    # Data integrity check
    # -------------------------------------------------
    print("Checking TRAIN data integrity ...\n")
    check_images_integrity(df_train, config.IMAGE_DIR)

    print("Checking VALIDATION data integrity ...\n")
    check_images_integrity(df_val, config.IMAGE_DIR)

    print("Checking TEST data integrity ...\n")
    check_images_integrity(df_test, config.IMAGE_DIR)