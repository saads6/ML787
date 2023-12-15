import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Loading the datasets
file_paths = [
    'C:/Users/saads/Documents/787_ML/data/diabetes_binary_health_indicators_BRFSS2015.csv',
    'C:/Users/saads/Documents/787_ML/data/diabetes_012_health_indicators_BRFSS2015.csv',
    'C:/Users/saads/Documents/787_ML/data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
]

# Function to prepare and split the dataset
def prepare_and_split_data(df, target_column, stratify_data=False):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    stratify = y if stratify_data else None
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

# Model initializations with hyperparameter grids for tuning
dt_classifier = DecisionTreeClassifier(random_state=42)
lr_classifier = LogisticRegression(random_state=42, max_iter=1000, multi_class='auto')
rf_classifier = RandomForestClassifier(random_state=42)

param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2']
}

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create GridSearchCV objects
dt_grid_search = GridSearchCV(dt_classifier, param_grid_dt, cv=5)
lr_grid_search = GridSearchCV(lr_classifier, param_grid_lr, cv=5)
rf_grid_search = GridSearchCV(rf_classifier, param_grid_rf, cv=5)

# Lists to store performance metrics, feature importances, ROC AUC scores,
# Precision-Recall AUC scores, and learning curves for each model
model_names = ["Decision Tree", "Logistic Regression", "Random Forest"]
models_to_evaluate = [dt_grid_search, lr_grid_search, rf_grid_search]

for file_path in file_paths:
    df = pd.read_csv(file_path)

    # Determine the target column based on the file name
    target_column = 'Diabetes_binary' if 'binary' in file_path else 'Diabetes_012'
    stratify_data = '5050split' not in file_path

    X_train, X_test, y_train, y_test = prepare_and_split_data(df, target_column, stratify_data)

    for model, model_name in zip(models_to_evaluate, model_names):
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print the model name, processing time, and results
        print(f"Results for {model_name}:")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
        print(f"Confusion Matrix:\n{conf_matrix}")

        # Plot feature importance
        if hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
            feature_importance = model.best_estimator_.feature_importances_
            plt.figure(figsize=(10, 5))
            sns.barplot(x=feature_importance, y=df.drop(target_column, axis=1).columns)
            plt.title(f'Feature Importance for {model_name}')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.show()

        # Plot ROC curve
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})', color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend(loc='lower right')
        plt.show()

        # Plot Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (Avg Precision = {avg_precision:.2f})', color='darkorange')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {model_name}')
        plt.legend(loc='lower left')
        plt.show()

        # Plot learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model.best_estimator_, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores_mean, label='Train', color='darkorange')
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2, color='darkorange')
        plt.plot(train_sizes, test_scores_mean, label='Validation', color='navy')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2, color='navy')
        plt.xlabel('Training Examples')
        plt.ylabel('Accuracy')
        plt.title(f'Learning Curve for {model_name}')
        plt.legend(loc='best')
        plt.show()
