import numpy as np
import torch

def create_dataset(df):
    sequences = df.astype(np.float32a).to_numpy().tolist()
    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    n_seq, seq_len, features = torch.stack(dataset).shape
    return dataset, seq_len, features
