import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
# Sample input/output sequences (tokens + tags)
sentences = [["The", "cat", "sat"],
             ["Dogs", "bark", "loudly"]]
tags = [["DET", "NOUN", "VERB"],
        ["NOUN", "VERB", "ADV"]]
 
# Build vocab
word2idx = {"<pad>": 0}
tag2idx = {"<pad>": -1}
for s in sentences:
    for w in s:
        word2idx.setdefault(w, len(word2idx))
for seq in tags:
    for t in seq:
        tag2idx.setdefault(t, len(tag2idx))
 
# Dataset
class SeqTagDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
 
    def __getitem__(self, idx):
        x = [word2idx[w] for w in self.X[idx]]
        y = [tag2idx[t] for t in self.Y[idx]]
        return torch.tensor(x), torch.tensor(y)
 
    def __len__(self):
        return len(self.X)
 
def collate_fn(batch):
    xs, ys = zip(*batch)
    x_pad = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    y_pad = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1)
    return x_pad, y_pad
 
loader = DataLoader(SeqTagDataset(sentences, tags), batch_size=2, collate_fn=collate_fn)
 
# Bi-LSTM model
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.bilstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, tagset_size)
 
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.bilstm(x)
        return self.fc(x)
 
model = BiLSTMTagger(len(word2idx), len(tag2idx))
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# Training loop
for epoch in range(5):
    for x, y in loader:
        out = model(x)
        out = out.view(-1, out.shape[-1])
        y = y.view(-1)
        loss = criterion(out, y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")