import pandas as pd
from models.ml_models import get_models
from train.ml_model_train import train_model
from metrics.metrics import classification_metrics,Roc_ConfusionMatrix

class Classification:
    
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train =  X_train.values
        self.Y_train =  Y_train.values
        self.Y_test  =   Y_test.values
        self.X_test  =    X_test.values
        self.models  =   get_models()
        self.metric_data = pd.DataFrame(columns=['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall'])
        self.pred_data = pd.DataFrame()

    def train_prediction(self, model_=None):
        if model_:
            model = self.models[model_]
            Y_pred = train_model(model, self.X_train, self.Y_train, self.X_test)
            acc, pre, f1_s, rec = classification_metrics(self.Y_test, Y_pred, True)

            # Append metrics to metric_data DataFrame
            self.metric_data.loc[i] = {
                'Model': model,
                'Accuracy': acc,
                'F1-Score': f1_s,
                'Precision': pre,
                'Recall': rec
            }

            # Create a column in pred_data with model name and y_pred values
            self.pred_data[model_ + '_y_pred'] = Y_pred
        else:
            for i, (model, _) in enumerate(self.models.items()):
                Y_pred = train_model(model, self.X_train, self.Y_train, self.X_test)
                acc, pre, f1_s, rec = classification_metrics(self.Y_test, Y_pred, True)

                # Append metrics to metric_data DataFrame
                self.metric_data.loc[i] = {
                    'Model': model,
                    'Accuracy': acc,
                    'F1-Score': f1_s,
                    'Precision': pre,
                    'Recall': rec
                }

                # Create a column in pred_data with model name and y_pred values
                self.pred_data[model + '_y_pred'] = Y_pred

    def best_model(self):
        # Sort the DataFrame based on 'Accuracy' column in descending order
        self.sorted_metric_data = self.metric_data.sort_values(by='Accuracy', ascending=False)

        # Find the maximum value in the 'Accuracy' column
        max_accuracy = self.sorted_metric_data['Accuracy'].max()

        # Highlight the maximum value in the 'Accuracy' column
        highlighted_metrics = self.sorted_metric_data.style.apply(
            lambda x: ['background-color: yellow' if x.name == 'Accuracy' and val == max_accuracy else '' for val in x],
            axis=0)
        print(f"Best Model : {self.models[self.sorted_metric_data['Model'].iloc[0]]}")
        return highlighted_metrics

    def metrics_plotter(self):
        
        Roc_ConfusionMatrix(self.Y_test, self.pred_data[self.sorted_metric_data['Model'].iloc[0]+'_y_pred'])
