import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
 
# Sample English-French pairs
pairs = [
    ("hello", "salut"),
    ("how are you", "comment ça va"),
    ("i am fine", "je vais bien"),
    ("thank you", "merci"),
    ("goodbye", "au revoir")
]
 
# Build vocab
def build_vocab(sentences):
    tokens = set(" ".join(sentences).split())
    word2idx = {word: i+2 for i, word in enumerate(tokens)}
    word2idx["<pad>"] = 0
    word2idx["<eos>"] = 1
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word
 
eng_sentences, fr_sentences = zip(*pairs)
eng2idx, idx2eng = build_vocab(eng_sentences)
fr2idx, idx2fr = build_vocab(fr_sentences)
 
# Tokenizer
def encode(sentence, vocab):
    return [vocab[word] for word in sentence.split()] + [vocab["<eos>"]]
 
# Dataset
class TranslationDataset(Dataset):
    def __init__(self, pairs):
        self.data = [(encode(e, eng2idx), encode(f, fr2idx)) for e, f in pairs]
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src), torch.tensor(tgt)
 
def collate_fn(batch):
    src, tgt = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src, batch_first=True)
    tgt = nn.utils.rnn.pad_sequence(tgt, batch_first=True)
    return src, tgt
 
loader = DataLoader(TranslationDataset(pairs), batch_size=2, collate_fn=collate_fn)
  
# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
 
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return hidden
 
# Decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
 
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden
 
# Model setup
INPUT_DIM = len(eng2idx)
OUTPUT_DIM = len(fr2idx)
ENC_EMB_DIM = DEC_EMB_DIM = 32
HIDDEN_DIM = 64
 
encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HIDDEN_DIM)
decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HIDDEN_DIM)
 
criterion = nn.CrossEntropyLoss(ignore_index=0)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
 
# Training loop
for epoch in range(20):
    for src, tgt in loader:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
 
        hidden = encoder(src)
        dec_input = tgt[:, :-1]
        target = tgt[:, 1:]
 
        outputs, _ = decoder(dec_input, hidden)
        loss = criterion(outputs.reshape(-1, OUTPUT_DIM), target.reshape(-1))
 
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")