from sklearn.metrics import confusion_matrix ,f1_score,precision_score,recall_score,accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 

from models.ml_models import get_models
def train_model(model, X_train: np.array, Y_train: np.array, X_test: np.array):
    if not all(isinstance(arr, np.ndarray) for arr in [X_train, Y_train, X_test]):
        raise TypeError('X_train, Y_train, X_test should be arrays')

    print(f'Initializing...: {model}')
    
    # Assuming get_models() is a function that returns a dictionary of models
    models_dict = get_models()

    if model not in models_dict:
        raise ValueError(f'Model {model} not found in the available models.')

    print('Training....')
    selected_model = models_dict[model]
    trained_model = selected_model.fit(X_train, Y_train)

    print('Testing.....')
    y_pred = trained_model.predict(X_test)

    return y_pred

    
def classification_metrics(y_test,y_pred,ret):
    acc=accuracy_score(y_test,y_pred)
    pre=precision_score(y_test,y_pred)
    f1_s=f1_score(y_test,y_pred)
    rec=recall_score(y_test,y_pred)
    #cm=confusion_matrix(y_test,y_pred)
    print(f'Accuracy   : {acc*100:.2f}')
    print(f'Precision  : {pre*100.:.2f}')
    print(f'F1 Score   : {f1_s*100:.2f}')
    print(f'Recall     : {rec*100:.2f}')
    if ret:
        return acc,pre,f1_s,rec
def Roc_ConfusionMatrix(y_test,y_pred,save=False):
    cm=confusion_matrix(y_test,y_pred)   
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True,fmt='d',cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Correct')
    if save:
        plt.savefig('Confusion_Matrix')
    plt.show()
    
    # Assuming y_true are your true labels, and y_scores are the predicted probabilities
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    if save:
        plt.savefig('ROC_curve')
    plt.show()