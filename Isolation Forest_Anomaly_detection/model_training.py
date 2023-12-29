import pandas as pd 
from sklearn.ensemble import IsolationForest
import numpy as np 
import matplotlib.pyplot as plt 
rn_seed=42
np.random.seed(rn_seed)

def target_remover(df):
    df.columns=list(range(df.shape[1]))
    return df.iloc[:,:-1]
def data_pre_model_train(normal_data,abnrmal_data):
    normal_data=target_remover(normal_data)
    abnormal_data=target_remover(abnormal_data)
    model=IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2),random_state=rn_seed)
    model.fit(normal_data.values)
    normal_data_score=model.predict(normal_data.values)
    correct=len([x for x in normal_data_score if x ==1])
    accuracy=correct/normal_data.shape[0]
    print(f'Accuracy : {accuracy*100}%')
def anomaly_score(model,abnormal_data,sample_size):
    abnormal_sample=abnormal_data[:sample_size]
    abnormal_sample.shape
    scores = {'abnormal_data_degree': [], 'abnormal_data_score': []}
    scores['abnormal_data_degree'].append(np.array(model.decision_function(abnormal_sample.values)))
    scores['abnormal_data_score'].append(np.array(model.predict(abnormal_sample.values)))
    scores = pd.DataFrame(scores)

    # Explode lists into separate rows
    scores_exploded = scores.apply(lambda col: col.explode(), axis=0)
    scores_exploded.reset_index(inplace=True)
    # Visualize the DataFrame
    scores_exploaded=scores_exploded.drop('index',axis=1)
    return scores_exploaded

def beat_plotter(abnormal_sample):
    plt.figure(figsize=(10,5))
    for i in range(5):
        target=abnormal_sample.loc[i,'iso_label']
        degree=abnormal_sample.loc[i,'iso_anomaly_degree']
        if target==1:
            label=f'Normal HB(degree={degree:.2f})'
            lw=1
        else:
            label=f'Abnormal(Anomaly)HB(degree={degree:.2f})'
            lw=2
        plt.plot(abnormal_sample.loc[i,:186],label=label,linewidth=lw)
    plt.title('Anomaly Detection Graph')
    plt.legend(loc='upper right')
    plt.xlabel('Heart Beat Features')
    plt.ylabel('Frequencies')
    plt.savefig('Isolation Forest Result')
    plt.show()
