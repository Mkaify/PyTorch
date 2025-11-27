import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
 
# Load and tokenize IMDB dataset
tokenizer = get_tokenizer("basic_english")
 
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)
 
# Build vocabulary from training data
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])
 
# Encode function
def encode(text):
    return vocab(tokenizer(text))
 
# Collate function for DataLoader
def collate_batch(batch):
    text_list, label_list = [], []
    for label, text in batch:
        text_tensor = torch.tensor(encode(text), dtype=torch.long)
        label_tensor = torch.tensor(1 if label == 'pos' else 0, dtype=torch.float)
        text_list.append(text_tensor)
        label_list.append(label_tensor)
    padded_texts = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    return padded_texts, torch.tensor(label_list)
 
# DataLoaders
train_iter = IMDB(split='train')
train_loader = DataLoader(list(train_iter), batch_size=16, shuffle=True, collate_fn=collate_batch)
 
# RNN-based classifier
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<pad>"])
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
 
    def forward(self, x):
        embeds = self.embedding(x)
        _, hidden = self.rnn(embeds)
        return torch.sigmoid(self.fc(hidden.squeeze(0)))
 
model = RNNClassifier(len(vocab), embed_dim=100, hidden_dim=128)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
# Training loop (1 epoch for brevity)
for epoch in range(1):
    for texts, labels in train_loader:
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")