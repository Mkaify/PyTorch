import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
# Sample NER-style dataset
sentences = [["John", "lives", "in", "New", "York"],
             ["Apple", "is", "a", "company"]]
labels = [["B-PER", "O", "O", "B-LOC", "I-LOC"],
          ["B-ORG", "O", "O", "O"]]
 
word2idx = {"<pad>": 0}
tag2idx = {"O": 0}
for s in sentences:
    for w in s:
        word2idx.setdefault(w, len(word2idx))
for seq in labels:
    for t in seq:
        tag2idx.setdefault(t, len(tag2idx))
 
# Dataset class
class NERDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
 
    def __len__(self):
        return len(self.sentences)
 
    def __getitem__(self, idx):
        x = [word2idx[w] for w in self.sentences[idx]]
        y = [tag2idx[t] for t in self.labels[idx]]
        return torch.tensor(x), torch.tensor(y)
 
# Collate function to pad sequences
def collate_fn(batch):
    xs, ys = zip(*batch)
    x_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    y_padded = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1)
    return x_padded, y_padded
 
dataset = NERDataset(sentences, labels)
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
 
# BiLSTM model
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128 * 2, tagset_size)
 
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        return self.fc(x)
 
model = BiLSTMTagger(len(word2idx), len(tag2idx))
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# Train
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