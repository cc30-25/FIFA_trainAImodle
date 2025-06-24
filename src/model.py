import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        """
        This defines the architecture of the attention layer.
        """
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)  # Linear layer for attention calculation

    def forward(self, lstm_out):
        attn_weights = self.attention(lstm_out)  # Calculate the attention weights
        attn_weights = F.softmax(attn_weights, dim=1)  # Apply softmax to get the attention probabilities
        context = torch.sum(attn_weights * lstm_out, dim=1)  # Calculate the context vector
        return context

class EventPredictorWithSeq(nn.Module):
    def __init__(self, static_input_dim, seq_input_dim, num_classes):
        """
        This defines the architecture of the model, which is two fully connected layers followed by an LSTM layer.
        """
        super(EventPredictorWithSeq, self).__init__()
        self.fc1 = nn.Linear(static_input_dim, 128) # Fully connected layer
        self.bn1 = nn.BatchNorm1d(128)  # BatchNorm layer
        self.dropout = nn.Dropout(config.DROPOUT)  # Dropout layer

        self.fc2 = nn.Linear(128, 64)   
        self.bn2 = nn.BatchNorm1d(64) 
        self.dropout2 = nn.Dropout(config.DROPOUT)  

        self.residual_proj = nn.Linear(128, 64)  # Projection layer for residual connection

        self.lstm = nn.LSTM(
            input_size = seq_input_dim, 
            hidden_size = 64, 
            num_layers = 2, 
            batch_first = True)  # LSTM layer

        self.attention_layer = AttentionLayer(hidden_dim=64)
        
        self.fc_fusion = nn.Linear(64 +64, 64)
        self.fc_out = nn.Linear(64, num_classes)  

    def forward(self, x_static, x_seq):
        # Forward pass for the static features
        static_out =  F.relu(self.bn1(self.fc1(x_static)))
        static_out = self.dropout(static_out)
        residual = static_out   # Save the output for residual connection
        static_out = F.relu(self.bn2(self.fc2(static_out)))
        static_out = self.dropout2(static_out)

        residual = self.residual_proj(residual)  # Project the residual to match the size of the output
        static_out = static_out + residual

        # LSTM pass
        lstm_out, (h_n, _) = self.lstm(x_seq)    # x_seq: (batch, sequence_length, seq_input_dim)
        # Use the attension layer on lstm_out
        lstm_out = self.attention_layer(lstm_out)
        

        # Concatenate the static and sequential features
        fusion = torch.cat([static_out, lstm_out], dim=1)
        fusion = F.relu(self.fc_fusion(fusion))
        logits = self.fc_out(fusion)
        return logits   # Logits are the raw scores output by the model
    
if __name__ == "__main__":
    temp_static_dim = 11
    temp_seq_dim = 11
    temp_num_classes = 9
    # Create an instance of the model
    model = EventPredictorWithSeq(temp_static_dim, temp_seq_dim, temp_num_classes)
    temp_static = torch.randn(8, temp_static_dim)  # 8 samples, 4 static features
    temp_seq = torch.randn(8, config.HISTORICAL_EVENT, temp_seq_dim)  # 8 samples, 10 events, 4 features per event
    output = model(temp_static, temp_seq)
    print("Model output shape: ", output.shape)