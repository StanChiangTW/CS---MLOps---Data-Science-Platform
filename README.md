# DSBA Platform

A toy MLOps Project

## Project Structure


- a `pyproject.toml` file contains the project metadata, including the dependencies. It is common to see a "setup.py" file in Python projects but we use this more modern approach to define the project metadata.
- The `src` folder contains the code code (dsba) as well as the code for the CLI, the API, the web app, the notebooks, as well as the Dockerfiles.
- The `tests` folder contains some unit and integration tests
- `.gitignore` is a special file name that will be detected by git. This file contains a list of files and folders that should not be committed to the repository. For example (see below for setup), the `.env` file is specific to your own deployment so it should not be committed to the repository (it may contain specific file paths that are only meaningful on your machine, and it may contain secrets like API keys - API keys and passwords should never be stored in a git repository).

## Installation (dev mode)

### Requirements

Your machine should have the following software installed:

- Python 3.12
- git
- to use the model training notebook (not required), you may need to install openmp (libomp) which is required by xgboost. But you can also not use the model_training module from this example or adapt it to use scikit-learn rather than xgboost.

### Clone the repository

- The first things to do is to copy this repository, to have a copy that you own on GitHub. This is because you are not allowed to push directly to the main repository owned by Joachim. Copying a repository on GitHub to have your own is called a "fork". You should understand that "forking" and "cloning" are not the same. Forking is a GitHub concept to copy a repository in your own GitHub account. Cloning basically means "downloading for the first time a repo to your computer". Just click on the fork button above when seeing this document from GitHub.

- Move into the folder you want to work in (I saw many students not choosing a folder and just working in their home directory, you don't want to do that)

- To be certain things are ok type:

```bash
git status
```

This should fail and tell you there is no repository at this location. I saw many students trying to clone a repository inside a repository, you also don't want to be in this situation.

Now you can clone the repository:

```bash
git clone <the address of your fork>
```

### Installing the project

cd into the repository folder.

Create a virtual environment with the following command (for windows, python, not python3).
Using the name ".venv" for your virtual environment is recommended.
It is quite standard and tools like vscode will automatically find it.

```bash
python3 -m venv .venv
```

Install dependencies (as specified in pyproject.toml):

```bash
pip install -e .
```

This will install the project in editable mode, meaning that any changes you make to the code will be reflected in your local environment.

## Running the tests

To run the tests, you can use the following command:

```bash
pytest
```

This will run all the tests in the `tests` folder.

## Usage

You must set the environment variable `DSBA_MODELS_ROOT_PATH` to the address you want to store the models in before you can use the platform.

For example as a MacOS user I set `/Users/joachim/dev/dsba/models_registry`.

There are many ways to set environment variables depending on the context.

In a python notebook, you can use the following code:

```python
import os
os.environ["DSBA_MODELS_ROOT_PATH"] = "/path/to/your/models"
```

In a terminal or shell script, you can use the following code (Linux and MacOS):

```bash
export DSBA_MODELS_ROOT_PATH="/path/to/your/models"
```

For windows, something of the sort may work:

```bash
set DSBA_MODELS_ROOT_PATH="C:\path\to\your\models"
```





# MLOps student project
This project aimed to create an interface for a bank employees that allows them to choose a prediction model, a customer dataset, and then provide a result on whether customers will churn or not.

To do this, we based our work on a machine learning project on bank churners that one of us had previously created. This project used the dataset of a kaggle challenge available at :  https://www.kaggle.com/datasets/thedevastator/predicting-credit-card-customer-attrition-with-m 


## Important Folders and Files

### Data

```bash
data/
```

This contains:
- the original `BankChurners.csv` file of the Kaggle challenge used for the original ML project
- the `X_test.csv` and the `y_test.csv` files created after the preprocessing of the original dataset, this files are the ones used for the rest of the MLOps project


### Models

```bash
models/
```

This contains the different models trained and available to predict results. The 4 models are :
- `lgbm_model.pkl`
- `rf_model.pkl`
- `svm_model.pkl`
- `xgb_model.pkl`


### Src

```bash
src/
```
This is the main source directory where the code resides.


#### API
An API is provided, it allows to interact with models. You can start the API by running:
```bash
uvicorn src.api.api:app --reload
```

Dockerized API
To run the API in a Docker container, follow these steps:
1. Build the Docker image:
```bash
docker build -t aryap25/mlops_docker:latest
```
2. Run the Docker container:
```bash
docker run -p 8000:8000 aryap25/mlops_docker:latest
```
The API will be available at http://127.0.0.1:8000/

Note: Ensure Docker is installed on your machine.


#### CLI

Not used in the MLOps project, normally it's used to list models registered on your system:

```bash
src/cli/dsba_cli list
```

Use a model to predict on a file:

```bash
src/cli/dsba_cli predict --input /path/to/your/data/file.csv --output /path/to/your/output/file.csv --model-id your_model_id
```


#### DSBA
This contains the core functionality of the MLOps project, including model handling, data preprocessing, and utilities for training and predicting.



#### Notebooks
This contains:
- `model_training_example.ipynb` the original example of Notebook of the MLOps platform project
- `Bank_MLOps.ipynb` the notebook of an ML project from which we used it for our MLOps project. The original code from the ML project has not been modified; it may contain some elements from LLM. To use the notebook, navigate to the `notebooks/` folder and open the file. You can use the provided utilities to train models, preprocess data, and evaluate performance.



## REMARKS
No remarks