import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# Simulated note sequences (MIDI note numbers 60 = C4, 62 = D4, etc.)
note_vocab = list(range(60, 73))  # C4 to B4
note2idx = {note: idx for idx, note in enumerate(note_vocab)}
idx2note = {idx: note for note, idx in note2idx.items()}

# Generate synthetic dataset of musical note sequences
class NoteDataset(Dataset):
    def __init__(self, num_sequences=1000, seq_len=20):
        self.data = []
        for _ in range(num_sequences):
            sequence = random.choices(note_vocab, k=seq_len + 1)
            x = [note2idx[n] for n in sequence[:-1]]
            y = [note2idx[n] for n in sequence[1:]]
            self.data.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Define RNN model
class MusicRNN(nn.Module):
    def __init__(self, vocab_size, emb_size=32, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        return self.fc(out)

# Training setup
dataset = NoteDataset()
loader = DataLoader(dataset, batch_size=32, shuffle=True)
model = MusicRNN(vocab_size=len(note_vocab))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train loop
for epoch in range(5):
    for x, y in loader:
        outputs = model(x)
        loss = criterion(outputs.view(-1, len(note_vocab)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Music generation function
def generate_sequence(model, start_note, length=30):
    model.eval()
    notes = [note2idx[start_note]]
    input_seq = torch.tensor(notes).unsqueeze(0)
    for _ in range(length):
        output = model(input_seq)
        next_note = torch.argmax(output[0, -1]).item()
        notes.append(next_note)
        input_seq = torch.tensor(notes).unsqueeze(0)
    return [idx2note[i] for i in notes]

# Generate a melody
start_note = 60  # C4
generated_notes = generate_sequence(model, start_note)
print("🎼 Generated Note Sequence:", generated_notes)