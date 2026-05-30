import os

# Directories
PROJECT_DIR = os.getcwd()           # Retrieve current directory
PROJECT_DIR = PROJECT_DIR[1:]       # Removed "/" at index 0 for clarity

DATA_DIR = "data"
RAW_DATA_DIR = f"{PROJECT_DIR}/{DATA_DIR}/raw"
IMAGE_DIR = f"{RAW_DATA_DIR}/crops/Crop_224_5fp_100K"

# Files
DF_TRAIN = f"/{RAW_DATA_DIR}/train.csv"
DF_TEST = f"/{RAW_DATA_DIR}/test_students.csv"
IMAGE_TEST = f"/{IMAGE_DIR}/database1/img00000048.webp"


# Parameters

params_train = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 4}

params_val = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 4}


if __name__ == "__main__":

    print("*****************************************************")
    print("config.py")
    print("*****************************************************")
    
    files = [DF_TRAIN,
            DF_TEST,
            IMAGE_TEST]

    for file in files:
        if os.path.exists(file):
            print(f"File O.k.! :\t{file}")
        else:
            print(f"File NOT O.k.! : {file}")
