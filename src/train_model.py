import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from src.data_loader import load_data
# from src.data_preprocess import preprocess_json_data
from src.data_preprocess import preprocess_sequential_data
from src.model import EventPredictorWithSeq
from src import config
from sklearn.model_selection import train_test_split

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
    def __init__(self, x_staic, x_seq, y):
        self.x_static = torch.tensor(x_staic, dtype=torch.float32)  # convert to tensor
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        # return the number of samples
        return len(self.x_static)
    
    def __getitem__(self, idx):
        return self.x_static[idx], self.x_seq[idx], self.y[idx]
    
def train():
    data = load_data()
    x_static, x_seq, y, result_map = preprocess_sequential_data(data)
    # x_static, x_seq, y, result_map = preprocess_json_data()
    unique, counts = np.unique(y, return_counts=True)
    print("Unique labels:", dict(zip(unique, counts)))


    # Split the data into training and validation sets
    x_static_train, x_static_val, x_seq_train, x_seq_val, y_train, y_val = train_test_split(
        x_static, x_seq, y, test_size=config.VALIDATION_SIZE, random_state=config.SEED
    )

    # Create datasets and dataloaders
    train_dataset = EventSequenceDataset(x_static_train, x_seq_train, y_train)
    val_dataset = EventSequenceDataset(x_static_val, x_seq_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    static_input_dim = x_static.shape[1]
    seq_input_dim = x_seq.shape[2]
    num_classes = len(result_map)

    # Initialize the model
    model = EventPredictorWithSeq(static_input_dim, seq_input_dim, num_classes)
    model.to(config.DEVICE)

    # Define the loss function and optimizer
    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / np.sum(weights) * len(counts)   # normalize
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for x_static_batch, x_seq_batch, y_batch in train_loader:
            x_static_batch, x_seq_batch, y_batch = x_static_batch.to(config.DEVICE), x_seq_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
            optimizer.zero_grad()
            output = model(x_static_batch, x_seq_batch)
            loss = criterion(output, y_batch)
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
                x_static_batch, x_seq_batch, y_batch = x_static_batch.to(config.DEVICE), x_seq_batch.to(config.DEVICE), y_batch.to(config.DEVICE)
                output = model(x_static_batch, x_seq_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item() * x_static_batch.size(0)

                _, predicted = torch.max(output.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                print("Sample probabilities:", torch.softmax(output, dim=1)[0].cpu().numpy())

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total

            print(f"Epoch [{epoch}/{config.NUM_EPOCHS}], Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), config.MODEL_NAME)
                print("Model saved")

    return model, result_map

if __name__ == "__main__":
    trained_model, result_map = train()