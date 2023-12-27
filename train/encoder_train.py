import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

#model_path = '/kaggle/working/model.pt'

def train_model(model, train_dataset, val_dataset, n_epochs=1, learning_rate=0.1, patience=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=patience, verbose=True)
    criterion = nn.L1Loss(reduction='sum')
    model_best_wts = copy.deepcopy(model.state_dict())
    delta = 0
    early_stopping = EarlyStopping(delta, model_path)
    best_loss = 10000
    history = {'train': [], 'val': []}

    for epoch in range(n_epochs):
        model = model.train()
        train_losses = []
        print('Epoch : ', epoch + 1)
        for seq_input in tqdm(train_dataset):
            optimizer.zero_grad()
            seq_input = seq_input.to(device)
            seq_pred = model(seq_input)
            loss = criterion(seq_pred, seq_input)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model = model.eval()
        val_losses = []

        with torch.no_grad():
            for val_input in tqdm(val_dataset):
                val_input = val_input.to(device)
                out = model(val_input)
                loss = criterion(out, val_input)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        print(f"Train Loss : {train_loss:.2f} | Val Loss : {val_loss:.2f}")

        if (epoch + 1) % 5 == 0:
            early_stopping(val_loss, model)

        scheduler.step(val_loss)  # Update learning rate based on validation loss

        if early_stopping.early_stop:
            print('Validation Loss is Not Decreasing (Early Stopping)')
            break

    model.load_state_dict(model_best_wts)
    return model.eval(), history
