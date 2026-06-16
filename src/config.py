import os
import torch

# ======================================================
# Directories and files
# ======================================================

# --- Main directories ---
LOCAL = True

if LOCAL:
    # --- Local ---
    PROJECT_DIR = os.getcwd()           # Retrieve current directory
    PROJECT_DIR = PROJECT_DIR[1:]       # Removed "/" at index 0 for clarity
else:
    # --- Telecom Cluster ---
    PROJECT_DIR = "home/infres/odelacruz-25/data-challenge-idemia"


# --- Models menu ---
MODELS =["MultiTaskMobileNet","MultiTaskResNet50"]
MODEL = MODELS[0]

# --- Dataset --- 
DATA_DIR = "data"
RAW_DATA_DIR = f"{PROJECT_DIR}/{DATA_DIR}"
IMAGE_DIR = f"{RAW_DATA_DIR}/crops/Crop_224_5fp_100K"
OUTPUT_DIR = f"{PROJECT_DIR}/outputs"

# --- Data files ---
DF_TRAIN = f"/{RAW_DATA_DIR}/occlusion_datasets/train.csv"
DF_TEST = f"/{RAW_DATA_DIR}/occlusion_datasets/test_students.csv"
IMAGE_TEST = f"/{IMAGE_DIR}/database1/img00000048.webp"

# ======================================================
# Data processing
# ======================================================

# --- Data integrity check ---
DATA_CHECK = False

# ======================================================
# Training
# ======================================================

# Device
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if USE_CUDA else "cpu")

LEARNING_RATE = 0.001
NUM_EPOCHS = 5

#GAMMA_GENDER = 0.1
#GAMMA_GENDER = 0.024
#GAMMA_GENDER = 0.015
GAMMA_GENDER = 0.00375

params_train = {'batch_size': 64,
          'shuffle': False,         # Set to False to use WeightedRandomSampler
          'num_workers': 4}

params_val = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 4}

# Include the epochs from which you want batch gender sampling counts
TARGET_EPOCHS = [1, 3]

# --- Data augmentation ---

# Random erasing
SCALE = [0.02, 0.10]
RATIO = [0.25, 0.7]
TRANSFORM_PROB = 0.30

# Gaussian blurr
SIGMA = [0.1, 2.0]
KERNEL_SZ = [5,5]

# ======================================================
# Information display
# ======================================================

# --- Model architecture ---
MODEL_INFO = False


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
