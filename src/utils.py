
from PIL import Image
from tqdm import tqdm
from src import config

def check_images_integrity(df,image_dir):

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            filename = df.loc[idx, 'filename']
            #print(filename)
            img2display = Image.open(f"/{image_dir}/{filename}")
        except ValueError as e:
            print(idx, e)




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

    # Split into train and validation
    df_val = df_train.loc[:20000].reset_index()
    df_train = df_train.loc[20000:].reset_index()

    # -------------------------------------------------
    # Data integrity check
    # -------------------------------------------------
    print("Checking TRAIN data integrity ...\n")
    check_images_integrity(df_train, config.IMAGE_DIR)

    print("Checking VALIDATION data integrity ...\n")
    check_images_integrity(df_val, config.IMAGE_DIR)

    print("Checking TEST data integrity ...\n")
    check_images_integrity(df_test, config.IMAGE_DIR)