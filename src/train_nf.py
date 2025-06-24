import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from src.data_loader import load_data
from src.data_preprocess import preprocess_sequential_data
from src.model import EventPredictorWithSeq
from src import config
import copy

result_map = {
    0: "team_a three pointer",
    1: "team_b three pointer",
    2: "team_a dunk",
    3: "team_b dunk",
    4: "team_a layup",
    5: "team_b layup",
    6: "team_a steal",
    7: "team_b steal",
    8: "miss"
}

class EventSequenceDataset(Dataset):
    def __init__(self, x_static, x_seq, y):
        self.x_static = torch.tensor(x_static, dtype=torch.float32)
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x_static)
    
    def __getitem__(self, idx):
        return self.x_static[idx], self.x_seq[idx], self.y[idx]

def train_fold(model, train_loader, val_loader, num_epochs):
    best_val_loss = float("inf")
    best_model_state = None
    criterion = nn.CrossEntropyLoss()  # you can add weighted loss here if needed
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for x_static_batch, x_seq_batch, y_batch in train_loader:
            x_static_batch = x_static_batch.to(config.DEVICE)
            x_seq_batch = x_seq_batch.to(config.DEVICE)
            y_batch = y_batch.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(x_static_batch, x_seq_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_static_batch.size(0)
        epoch_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_static_batch, x_seq_batch, y_batch in val_loader:
                x_static_batch = x_static_batch.to(config.DEVICE)
                x_seq_batch = x_seq_batch.to(config.DEVICE)
                y_batch = y_batch.to(config.DEVICE)
                outputs = model(x_static_batch, x_seq_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * x_static_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        print(f"Epoch {epoch}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_state)
    return model

def cross_validation_training(x_static, x_seq, y, num_folds=5):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config.SEED)
    ensemble_models = []
    fold = 0
    for train_index, val_index in skf.split(x_static, y):
        fold += 1
        print(f"Training fold {fold}/{num_folds}")
        X_static_train, X_static_val = x_static[train_index], x_static[val_index]
        X_seq_train, X_seq_val = x_seq[train_index], x_seq[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        train_dataset = EventSequenceDataset(X_static_train, X_seq_train, y_train)
        val_dataset = EventSequenceDataset(X_static_val, X_seq_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        model = EventPredictorWithSeq(x_static.shape[1], x_seq.shape[2], len(result_map))
        model.to(config.DEVICE)
        
        trained_model = train_fold(model, train_loader, val_loader, config.NUM_EPOCHS)
        ensemble_models.append(trained_model)
    return ensemble_models

def train():
    data = load_data()
    x_static, x_seq, y, result_map = preprocess_sequential_data(data)
    unique, counts = np.unique(y, return_counts=True)
    print("Unique labels:", dict(zip(unique, counts)))
    
    # Instead of a single train/test split, use cross-validation
    ensemble_models = cross_validation_training(x_static, x_seq, y, num_folds=5)
    
    # Optionally, you can save the ensemble models individually.
    # For example:
    for idx, model in enumerate(ensemble_models):
        torch.save(model.state_dict(), f"{config.MODEL_PATH}_fold{idx}.pth")
    
    return ensemble_models, result_map

if __name__ == "__main__":
    ensemble_models, result_map = train()
