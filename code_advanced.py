import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Trains and evaluates the given model.
    Args:
    - model: The machine learning model to train and evaluate.
    - X_train, X_test, y_train, y_test: The training and testing data.
    - model_name: The name of the model (for reporting purposes).

    Returns:
    - accuracy: The accuracy of the model on the test set.
    - report: The classification report.
    - conf_matrix: The confusion matrix.
    - feature_importance: The importance of features (if applicable).
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Extracting feature importance if possible
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = model.coef_[0]

    # Print the model name and results if model_name is provided
    if model_name:
        print(f"Results for {model_name}:")
        print(f"Accuracy: {accuracy}")
        print(f"Classification Report:\n{report}")
        print(f"Confusion Matrix:\n{conf_matrix}")

    return accuracy, report, conf_matrix, feature_importance

# Function to prepare and split the dataset
def prepare_and_split_data(df, target_column, stratify_data=False):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    stratify = y if stratify_data else None
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

# Loading the datasets
file_paths = [
    # Paths to the datasets
    'C:/Users/saads/Documents/787_ML/data/diabetes_binary_health_indicators_BRFSS2015.csv',
    'C:/Users/saads/Documents/787_ML/data/diabetes_012_health_indicators_BRFSS2015.csv',
    'C:/Users/saads/Documents/787_ML/data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv'
]
datasets = [pd.read_csv(fp) for fp in file_paths]


# Model initializations
dt_classifier = DecisionTreeClassifier(random_state=42)
lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
rf_classifier = RandomForestClassifier(random_state=42)
svm_classifier = SVC(kernel='linear', random_state=42)

# Iterating over datasets and models
for i, df in enumerate(datasets):
    target_column = 'Diabetes_binary' if 'binary' in file_paths[i] else 'Diabetes_012'
    stratify_data = '5050split' not in file_paths[i]

    X_train, X_test, y_train, y_test = prepare_and_split_data(df, target_column, stratify_data)

    # Train and evaluate Decision Tree
    train_evaluate_model(dt_classifier, X_train, X_test, y_train, y_test, "Decision Tree")

    # Train and evaluate Logistic Regression
    train_evaluate_model(lr_classifier, X_train, X_test, y_train, y_test, "Logistic Regression")

    # Train and evaluate Random Forest
    train_evaluate_model(rf_classifier, X_train, X_test, y_train, y_test, "Random Forest")

    # Train and evaluate SVM
    train_evaluate_model(svm_classifier, X_train, X_test, y_train, y_test, "SVM")

# Training and evaluating models
model_performance = {}
for name, model in models.items():
    accuracy, report, conf_matrix, feature_importance = train_evaluate_model(model, X_train, X_test, y_train, y_test)
    model_performance[name] = {
        "accuracy": accuracy,
        "feature_importance": feature_importance,
        "conf_matrix": conf_matrix
    }
# Plotting model accuracies
plt.figure(figsize=(10, 6))
sns.barplot(x=list(model_performance.keys()), y=[m['accuracy'] for m in model_performance.values()])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

# Plotting feature importance (for models that support it)
for name, performance in model_performance.items():
    if performance['feature_importance'] is not None:
        plt.figure(figsize=(12, 8))
        importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': performance['feature_importance']})
        sns.barplot(x='Importance', y='Feature', data=importance_df.sort_values(by='Importance', ascending=False))
        plt.title(f'Feature Importance for {name}')
        plt.show()

# Define the parameter grid for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs']  # Algorithm for optimization
}

# Create a GridSearchCV object
grid_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train, y_train)

# Best parameters and best score
print("Best Parameters for Logistic Regression:", grid_lr.best_params_)
print("Best Score for Logistic Regression:", grid_lr.best_score_)



# Assuming we're using a model like RandomForest
importances = grid_rf.best_estimator_.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Plotting model accuracies
plt.figure(figsize=(10, 6))
sns.barplot(x=list(model_performance.keys()), y=[m['accuracy'] for m in model_performance.values()])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.show()

# Plotting feature importance for models that support it
for name, performance in model_performance.items():
    if performance['feature_importance'] is not None:
        plt.figure(figsize=(12, 8))
        importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': performance['feature_importance']})
        sns.barplot(x='Importance', y='Feature', data=importance_df.sort_values(by='Importance', ascending=False))
        plt.title(f'Feature Importance for {name}')
        plt.show()

# Example of hyperparameter grid definition and grid search for Logistic Regression
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs']  # Algorithm for optimization
}

# Create a GridSearchCV object for Logistic Regression
grid_lr = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train, y_train)

# Best parameters and best score for Logistic Regression
print("Best Parameters for Logistic Regression:", grid_lr.best_params_)
print("Best Score for Logistic Regression:", grid_lr.best_score_)

plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()





