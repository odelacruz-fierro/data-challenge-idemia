import os
import torch

# ======================================================
# Database
# ======================================================

# --- Main directories ---
PROJECT_DIR = os.getcwd()           # Retrieve current directory
PROJECT_DIR = PROJECT_DIR[1:]       # Removed "/" at index 0 for clarity
DATA_DIR = "data"
RAW_DATA_DIR = f"{PROJECT_DIR}/{DATA_DIR}"
IMAGE_DIR = f"{RAW_DATA_DIR}/crops/Crop_224_5fp_100K"
OUOTPUT_DIR = f"{PROJECT_DIR}/outputs"

# --- Data files ---
DF_TRAIN = f"/{RAW_DATA_DIR}/occlusion_datasets/train.csv"
DF_TEST = f"/{RAW_DATA_DIR}/occlusion_datasets/test_students.csv"
IMAGE_TEST = f"/{IMAGE_DIR}/database1/img00000048.webp"

VAL_PREDICTIONS = f"/{OUOTPUT_DIR}/best_validation_predictions.csv"
TRAIN_HISTORY = f"/{OUOTPUT_DIR}/training_history.csv"

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


# Training
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

#GAMMA_GENDER = 0.1
#GAMMA_GENDER = 0.024
GAMMA_GENDER = 0.015


# Data augmentation
#SCALE = [0.08, 0.15]
SCALE = [0.02, 0.10] # Ancien : [0.08, 0.15]

RATIO = [0.25, 0.7]

#TRANSFORM_PROB = 0.5
TRANSFORM_PROB = 0.30 # Ancien : 0.5




params_train = {'batch_size': 64,
          'shuffle': False,         # Set to False to use WeightedRandomSampler
          'num_workers': 4}

params_val = {'batch_size': 64,
          'shuffle': False,
          'num_workers': 4}


# ======================================================
# Information display
# ======================================================

# --- Model architecture ---
MODEL_INFO = False


# ======================================================
# Outputs
# ======================================================



VALIDATION_RESULTS = True
TEST_RESULTS = False



if __name__ == "__main__":

    print("*****************************************************")
    print("config.py")
    print("*****************************************************")


    files = [DF_TRAIN,
            DF_TEST,
            IMAGE_TEST,
            VAL_PREDICTIONS,
            TRAIN_HISTORY]

    for file in files:
        if os.path.exists(file):
            print(f"File O.k.! :\t{file}")
        else:
            print(f"File NOT O.k.! : {file}")
