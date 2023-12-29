import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def data_splitting(abdf,ndf):

    rn_seed=42
    np.random.seed(rn_seed)
    ab_sampledata=abdf.sample(frac=0.385,random_state=rn_seed)
    final_data=pd.concat([ndf,ab_sampledata],axis=0,ignore_index=True)
    final_dataset=final_data.sample(frac=1).reset_index(drop=True)
    
    test_ratio=0.2
    train_val_ratio=0.8
    test_ratio_size=int(final_dataset.shape[0]*test_ratio)
    train_val_ratio=int(final_data.shape[0]-test_ratio_size)
    train_val_ratio,test_ratio_size

    train_val_data,test_data=final_dataset[:train_val_ratio],final_dataset[train_val_ratio:]

    target=train_val_data.iloc[:,-1]
    train_val_data=train_val_data.iloc[:,:-1]
    
    X_train, X_test, y_train, y_test = train_test_split(train_val_data,target,test_size=0.2, random_state=rn_seed)

    return train_val_data,test_data,X_train,y_train,y_test