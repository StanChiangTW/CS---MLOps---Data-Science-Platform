from dataclasses import dataclass
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    make_scorer
)
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
import logging
from typing import Any
from sklearn.model_selection import cross_val_score 


@dataclass
class ClassifierEvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: Any = None

def get_cross_val_score(
    fitted_model: ClassifierMixin, X: pd.DataFrame, y: pd.Series, num_cv: int = 5
) -> ClassifierEvaluationResult:
    
    f1_scorer=make_scorer(f1_score,average='weighted', zero_division=0) #average='weighted' ensures that the F1-score is computed for each class separately and then weighted by the number of true instances for each class
    f1=cross_val_score(fitted_model, X, y, cv=num_cv, scoring=f1_scorer)

    #PRECISION (proportion of correctly predicted positive observations out of all predicted positives)
    precision_scorer=make_scorer(precision_score,average='weighted', zero_division=0) 
    precision=cross_val_score(fitted_model, X , y ,cv=num_cv,scoring=precision_scorer)

    #RECALL (proportion of correctly predicted positive observations out of all actual positives)
    recall_scorer=make_scorer(recall_score,average='weighted', zero_division=0)
    recall=cross_val_score(fitted_model, X , y ,cv=num_cv,scoring=recall_scorer)

    #ACCURACY (proportion of correctly classified observations out of the total number of observations)
    accuracy= cross_val_score(fitted_model, X , y ,cv=num_cv, scoring='accuracy')

    return ClassifierEvaluationResult(
        accuracy= mean(accuracy),
        precision= mean(precision),
        recall= mean(recall),
        f1_score= mean(f1)
    )

def get_test_score(predictions, y_test) -> ClassifierEvaluationResult:
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    return ClassifierEvaluationResult(
        accuracy = accuracy,
        precision = precision,  
        recall = recall,
        f1_score = f1,
    )

def add_to_results(df, model_name, cv_metrics, test_metrics):
    logging.info(f"Storing results for {model_name}")
    
    # Convert to dictionary if not already a dictionary
    cv_dict = cv_metrics.__dict__ if hasattr(cv_metrics, '__dict__') else cv_metrics
    test_dict = test_metrics.__dict__ if hasattr(test_metrics, '__dict__') else test_metrics
    
    new_rows = pd.DataFrame([
        {"Dataset": "Cross-Validation", "Model": model_name, **cv_dict},
        {"Dataset": "Test Set", "Model": model_name, **test_dict}
    ])
    
    return pd.concat([df, new_rows], ignore_index=True)

def plot_model_comparison(results_df: pd.DataFrame, y_axis_start: float = 0.4):
    # Set style for Seaborn plots
    logging.info("Generating model comparison plots..")
    sns.set_style("whitegrid")
    
    # Plot Accuracy, Precision, Recall, F1-Score, and FOR for each model
    metrics_plot = ["accuracy", "precision", "recall", "f1_score"]
    metrics_titles = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    # Set up the figure for subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, (metric, metric_title) in enumerate(zip(metrics_plot, metrics_titles)):
        sns.barplot(
            data=results_df,
            x="Model",
            y=metric,
            hue="Dataset",
            ax=axes[i],
            palette="muted"
        )
        axes[i].set_title(f"Model Comparison: {metric}")
        axes[i].set_ylabel(metric)
        axes[i].set_xlabel("Model")
        axes[i].legend(title="Dataset", loc="lower right")
    
        axes[i].set_ylim(y_axis_start, axes[i].get_ylim()[1]) #here to adjust vertical axis : 0; 0,4 or other value
    
    # Adjust spacing between plots
    plt.tight_layout()
    
    return fig


def evaluate_models(models, X_train, y_train, X_test, y_test):

    results_df = pd.DataFrame(columns=["Dataset", "Model", "accuracy", "precision", "recall", "f1_score"])

    for model_name, model in models.items():
        cv_metrics = get_cross_val_score(model, X_train, y_train)
        y_pred_test = model.predict(X_test)
        test_metrics = get_test_score(y_test, y_pred_test)
        results_df = add_to_results(results_df, model_name, cv_metrics, test_metrics)

    results_df = results_df.sort_values(by=["Dataset"], ignore_index=True)

    
    
    

    return results_df, plot_model_comparison(results_df)


#models = {
#    "SVM": best_model_svm,
#    "Random Forest": best_model_rfc,
#    "XGBoost": best_model_xgb,
#    "LGBM": best_model_lgbm
#}
#results_df = evaluate_models(models, X_train, y_train, X_test, y_test)

