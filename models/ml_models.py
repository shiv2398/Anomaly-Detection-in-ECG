from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

def get_models():
    return {
        'lr_classifier': LogisticRegression(max_iter=5000, C=1.0),  # Example hyperparameter (C) for Logistic Regression
        'knn_classifier': KNeighborsClassifier(n_neighbors=5),  # Example hyperparameter (n_neighbors) for KNN
        'dt_classifier': DecisionTreeClassifier(max_depth=10),  # Example hyperparameter (max_depth) for Decision Tree
        'svc_classifier': SVC(C=1.0, kernel='rbf'),  # Example hyperparameters (C, kernel) for SVM
        'rf_classifier': RandomForestClassifier(n_estimators=100),  # Example hyperparameter (n_estimators) for Random Forest
        'gb_classifier': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1),  # Example hyperparameters (n_estimators, learning_rate) for Gradient Boosting
        'mlp_classifier': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000),  # Example hyperparameters (hidden_layer_sizes, max_iter) for MLP
        'ada_classifier': AdaBoostClassifier(n_estimators=50, learning_rate=1.0),  # Example hyperparameters (n_estimators, learning_rate) for AdaBoost
        'bagging_classifier': BaggingClassifier(n_estimators=50),  # Example hyperparameter (n_estimators) for Bagging
    }

# Call the function to get the dictionary of classifiers with updated hyperparameters
classifiers = get_models()

