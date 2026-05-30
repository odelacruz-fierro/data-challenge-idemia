

import pandas as pd
from src import config
from src import utils as tools



def main():
    print()
    print("*****************************************************")
    print("#")
    print("#")
    print("#")    
    print("# Welcome to Data IADATA704 data challenge pipeline")
    print("# Powered By Idemia and Télécom Paris")
    print("#")
    print("# Pipeline created by Oscar DE LA CRUZ")
    print("#")
    print("*****************************************************\n")

    print(f"Loading data from: {config.RAW_DATA_DIR}\n")

    # Datasets
    df_train = pd.read_csv(config.DF_TRAIN, delimiter=',')
    df_test = pd.read_csv(config.DF_TEST, delimiter=',')

    # Remove nan values
    df_train = df_train.dropna()
    df_test = df_test.dropna()
    
    # Data split
    df_val = df_train.loc[:20000].reset_index()
    df_train = df_train.loc[20000:].reset_index()

    # Data integrity check
    print("\nChecking TRAIN data integrity ...")
    tools.check_images_integrity(df_train, config.IMAGE_DIR)

    print("\nChecking VALIDATION data integrity ...")
    tools.check_images_integrity(df_val, config.IMAGE_DIR)

    print("\nChecking TEST data integrity ...")
    tools.check_images_integrity(df_test, config.IMAGE_DIR)

if __name__ == "__main__":
    main()