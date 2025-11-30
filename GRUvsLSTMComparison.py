import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import random
 
# Toy dataset: binary classification of digit sequences
sequences = [[1, 2, 3], [4, 5], [1, 1, 1], [5, 4, 3], [2, 2], [0, 1, 0]]
labels = [1, 0, 1, 0, 1, 0]  # Arbitrary class labels
 
# Vocabulary
vocab = {x: x for x in range(6)}  # integers 0–5
 
# Dataset class
class SequenceDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
 
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])
 
    def __len__(self):
        return len(self.X)
 
def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_pad = pad_sequence(xs, batch_first=True, padding_value=0)
    ys = torch.tensor(ys)
    return xs_pad, ys
 
loader = DataLoader(SequenceDataset(sequences, labels), batch_size=2, shuffle=True, collate_fn=collate_fn)
 
# Shared base model structure
class BaseRNN(nn.Module):
    def __init__(self, mode='lstm'):
        super().__init__()
        self.embedding = nn.Embedding(6, 16)
        if mode == 'lstm':
            self.rnn = nn.LSTM(16, 32, batch_first=True)
        elif mode == 'gru':
            self.rnn = nn.GRU(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 2)
 
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])  # use last hidden state
 
# Instantiate models
lstm_model = BaseRNN(mode='lstm')
gru_model = BaseRNN(mode='gru')
 
# Training function
def train_model(model, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    print(f"Training {name}")
    for epoch in range(5):
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"{name} Epoch {epoch+1}, Loss: {loss.item():.4f}")
    print()
 
# Train both models
train_model(lstm_model, "LSTM")
train_model(gru_model, "GRU")