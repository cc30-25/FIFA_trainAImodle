import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.data_loader import load_data
from src.data_preprocess import preprocess_sequential_data
from src.model import EventPredictorWithSeq
from src import config
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class BasketballSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, x_static, x_seq, y):
        self.x_static = torch.tensor(x_static, dtype=torch.float32)
        self.x_seq = torch.tensor(x_seq, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x_static)

    def __getitem__(self, idx):
        return self.x_static[idx], self.x_seq[idx], self.y[idx]
    
def evaluate():
    data = load_data()
    x_static, x_seq, y, result_map = preprocess_sequential_data(data)

    # Split the data into training and validation sets
    _, x_static_test, _, x_seq_test, _, y_test = train_test_split(
        x_static, x_seq, y, test_size=config.VALIDATION_SIZE, random_state=config.SEED
    )

    test_dataset = BasketballSequenceDataset(x_static_test, x_seq_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    static_input_dim = x_static_test.shape[1]
    seq_input_dim = x_seq_test.shape[2]
    num_classes = len(np.unique(y_test))

    model = EventPredictorWithSeq(static_input_dim, seq_input_dim, num_classes)
    model.load_state_dict(torch.load(config.MODEL_NAME, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_static_batch, x_seq_batch, y_batch in test_loader:
            x_static_batch = x_static_batch.to(config.DEVICE)
            x_seq_batch = x_seq_batch.to(config.DEVICE)
            outputs = model(x_static_batch, x_seq_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    print(classification_report(all_labels, all_preds, target_names=result_map.keys()))

if __name__ == "__main__":
    evaluate()

