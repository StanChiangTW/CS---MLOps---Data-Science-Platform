"""
This module is just a convenience to train a simple classifier.
Its presence is a bit artificial for the exercice and not required to develop an MLOps platform.
The MLOps course is not about model training.
"""

from dataclasses import dataclass
import logging
import pandas as pd
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.base import ClassifierMixin
from model_registry import ClassifierMetadata

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# Training function for SVM Classifier
def train_svm_classifier(
    X = pd.DataFrame, y = pd.Series, model_id: str = None
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start training a SVM Classfier with hyperparameter tuning")

    # Define base model
    model = SVC(random_state=42)

    param_grid = {
        'C': [0.1, 2],                # Smaller range for regularization
        'kernel': ['linear', 'rbf'],      # Focusing on simpler kernels
        'gamma': ['scale', 1],       # Reduced options for RBF kernel
        'class_weight': [None, 'balanced']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        verbose=1
    )
    grid_search.fit(X, y)

    # Best model
    best_model = grid_search.best_estimator_
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    logging.info(f"Best F1 Score: {grid_search.best_score_:.2f}")

    best_model.fit(X, y)

    logging.info("Done training SVM classifier")

    # Metadata
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="SVM",
        target_column=y.name,
        hyperparameters=grid_search.best_params_,
        description="SVM with GridSearchCV",
        performance_metrics={"f1_score": grid_search.best_score_},
    )

    return best_model, metadata



# Training function for Random Forest Classifier
def train_rfc_classifier(
    X: pd.DataFrame, y: pd.Series, model_id: str = None
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start training a random forest classifier with hyperparameter tuning")

    # Define base model
    model = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [7,10,12],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': ['sqrt'],
        'bootstrap': [True]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',  # Metric to optimize
        cv=5,                # Number of folds
        verbose=1             #Verbosity of printing messages. Valid values of 0 (silent), 1 (warning), 2 (info), and 3 (debug)
    )
    grid_search.fit(X, y)

    # Best model
    best_model = grid_search.best_estimator_
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    logging.info(f"Best F1 Score: {grid_search.best_score_:.2f}")

    best_model.fit(X, y)

    logging.info("Done training random forest classifier")

    # Metadata
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="random_forest",
        target_column=y.name,
        hyperparameters=grid_search.best_params_,
        description="Random forest with GridSearchCV",
        performance_metrics={"f1_score": grid_search.best_score_},
    )

    return best_model, metadata


# Training function for XGBoost Classifier
def train_xgboost_classifier(
    X: pd.DataFrame, y: pd.Series, model_id: str = None
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start training a xgboost classifier with hyperparameter tuning")

    # Define base model
    model = xgb.XGBClassifier(random_state=42)

    # Define grid
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [2, 3],
        'n_estimators': [50, 100, 150]
    }

    # Grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        verbose=1
    )
    grid_search.fit(X, y)

    # Best model
    best_model = grid_search.best_estimator_
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    logging.info(f"Best F1 Score: {grid_search.best_score_:.2f}")

    best_model.fit(X, y)

    logging.info("Done training xgboost classifier")

    # Metadata
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="xgboost",
        target_column=y.name,
        hyperparameters=grid_search.best_params_,
        description="XGBoost with GridSearchCV",
        performance_metrics={"f1_score": grid_search.best_score_},
    )

    return best_model, metadata


# Training function for LightGBM Classifier
def train_lgbm_classifier(
    X: pd.DataFrame, y: pd.Series, model_id: str = None
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    logging.info("Start training a xgboost classifier with hyperparameter tuning")

    model = lgb.LGBMClassifier(random_state=42, verbose=-1)

    param_grid = {
        'learning_rate': [0.2, 0.3, 0.4],
        'max_depth': [2,5],
        'n_estimators': [100,200],
        'min_child_weight': [2,5],
        'gamma': [1],
        'reg_lambda': [1],
        'reg_alpha': [1]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        verbose=1
    )
    grid_search.fit(X, y)

    # Best model
    best_model = grid_search.best_estimator_
    logging.info(f"Best Parameters: {grid_search.best_params_}")
    logging.info(f"Best F1 Score: {grid_search.best_score_:.2f}")

    best_model.fit(X, y)

    logging.info("Done training lgbm classifier")

    # Metadata
    metadata = ClassifierMetadata(
        id=model_id,
        created_at=str(datetime.now()),
        algorithm="lgbm",
        target_column=y.name,
        hyperparameters=grid_search.best_params_,
        description="LGBM with GridSearchCV",
        performance_metrics={"f1_score": grid_search.best_score_},
    )

    return best_model, metadata


