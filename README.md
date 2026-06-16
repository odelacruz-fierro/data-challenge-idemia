# **First approach - CNN parameter tuning**

This pipelien implements mainly a ResNet50 CNN modified to perform multi-task learning.

The instructions below assume you have installed `uv` project manager.

You will be able to run both the training and test pipelines.


## Training script `train.py`

Before running this script make sure you have created a directory called `outputs` at the project's root directory.<br>

For example if your project's directory is `data-challenge`, then you will have:

```console
data-challenge/outputs
```

In order to execute the `train.py` please follow the following command line structure.

In the terminal, you **must** execute it from the project root directory.

```console
uv run python -m train
```

Once you obtain a model, create a folder `RUN_XX` inside the `outputs` folder and place the generated files inside it.

For example:

```console
data-challenge/outputs/RUN_XX/

    best_model.pth
    best_validation_predictions.csv
    epoch_1_gender_dist_aug.csv
    training_history.csv

```

## Test script `test.py`

In order to execute the `test.py` please follow the following command line struccture.

In the terminal, you **must** execute it from toe project root directory.

```console
uv run python -m test

```

Since you already created a `RUN_XX/` folder, you will have to choose this folder when prompted by the script.<br>
This way the `test_predictions.csv` will be automatically placed in that folder.

```console
data-challenge/outputs/RUN_XX/

    best_model.pth
    best_validation_predictions.csv
    epoch_1_gender_dist_aug.csv
    training_history.csv
    test_predictions.csv
```
