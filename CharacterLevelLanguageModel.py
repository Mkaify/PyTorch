import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
# Sample text for demonstration
text = "hello world this is a simple character level language model"
 
# Create vocabulary of characters
chars = sorted(list(set(text)))
char2idx = {ch: i for i, ch in enumerate(chars)}
idx2char = {i: ch for ch, i in char2idx.items()}
 
# Convert text to indices
sequence = [char2idx[c] for c in text]
 
# Hyperparameters
seq_len = 10
batch_size = 4
 
# Dataset of character sequences
class CharDataset(Dataset):
    def __init__(self, sequence, seq_len):
        self.data = []
        for i in range(len(sequence) - seq_len):
            self.data.append((sequence[i:i+seq_len], sequence[i+seq_len]))
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)
 
dataset = CharDataset(sequence, seq_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
 
# RNN model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
 
    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        out = self.fc(out[:, -1, :])  # predict last character
        return out, h
 
model = CharRNN(len(chars))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
 
# Training loop
for epoch in range(10):
    for x, y in loader:
        out, _ = model(x)
        loss = criterion(out, y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
 
# Sampling function
def generate_text(model, start_seq, length=50):
    model.eval()
    input_seq = torch.tensor([char2idx[c] for c in start_seq]).unsqueeze(0)
    h = None
    generated = start_seq
    for _ in range(length):
        out, h = model(input_seq, h)
        next_idx = torch.argmax(out[:, -1], dim=1).item()
        next_char = idx2char[next_idx]
        generated += next_char
        input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[next_idx]])], dim=1)
    return generated
 
# Example text generation
print("\nGenerated text:")
print(generate_text(model, start_seq="hello "))