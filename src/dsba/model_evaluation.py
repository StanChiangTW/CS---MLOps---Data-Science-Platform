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
)
import matplotlib.pyplot as plt
import seaborn as sns
from dsba.preprocessing import preprocess_dataframe, split_features_and_target
from statistics import mean


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


def evaluate_models(models, X_train, y_train, X_test, y_test):
    """Evaluates multiple models and stores results in a DataFrame."""
    results_df = pd.DataFrame(columns=["Dataset", "Model", "accuracy", "precision", "recall", "f1_score"])

    for model_name, model in models.items():
        cv_metrics = get_cross_val_score(model, X_train, y_train)
        y_pred_test = model.predict(X_test)
        test_metrics = get_test_score(y_test, y_pred_test)
        results_df = add_to_results(results_df, model_name, cv_metrics, test_metrics)

    results_df = results_df.sort_values(by=["Dataset"], ignore_index=True)

    # Display and plot results
    print(results_df)
    # plot_model_comparison(results_df)

    return results_df






'''
def evaluate_classifier(
    classifier: ClassifierMixin, target_column: str, df: pd.DataFrame
) -> ClassifierEvaluationResult:
    df = preprocess_dataframe(df)
    X, y_actual = split_features_and_target(df, target_column)
    y_predicted = classifier.predict(X)

    accuracy = accuracy_score(y_actual, y_predicted)
    precision = precision_score(
        y_actual, y_predicted, average="weighted", zero_division=0
    )
    recall = recall_score(y_actual, y_predicted, average="weighted", zero_division=0)
    f1 = f1_score(y_actual, y_predicted, average="weighted", zero_division=0)
    conf_matrix = confusion_matrix(y_actual, y_predicted)

    return ClassifierEvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        confusion_matrix=conf_matrix,
    )
'''

# Display functions :

'''
def visualize_classification_evaluation(result: ClassifierEvaluationResult):
    confusion_maxtrix_fig = plot_confusion_matrix(result)
    plt.show(confusion_maxtrix_fig)
    evaluation_metrics_fig = plot_classification_metrics(result)
    plt.show(evaluation_metrics_fig)


def plot_confusion_matrix(result: ClassifierEvaluationResult) -> plt.Figure:
    confusion_matrix_df = pd.DataFrame(result.confusion_matrix)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(confusion_matrix_df, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig


def plot_classification_metrics(result: ClassifierEvaluationResult) -> plt.Figure:
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    scores = [result.accuracy, result.precision, result.recall, result.f1_score]
    metric_data = pd.DataFrame({"Metric": metrics, "Score": scores})
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.barplot(data=metric_data, x="Metric", y="Score", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Classification Metrics")
    ax.set_ylabel("Score")
    return fig
