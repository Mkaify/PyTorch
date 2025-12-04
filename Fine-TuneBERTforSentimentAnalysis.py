from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
 
# Load IMDb dataset
dataset = load_dataset("imdb")
train_data = dataset["train"].shuffle(seed=42).select(range(2000))  # Subset for quick training
test_data = dataset["test"].select(range(500))
 
# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
 
# Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
 
train_tokens = train_data.map(tokenize_function, batched=True)
test_tokens = test_data.map(tokenize_function, batched=True)
 
# Convert to PyTorch
train_tokens.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_tokens.set_format("torch", columns=["input_ids", "attention_mask", "label"])
 
train_loader = DataLoader(train_tokens, batch_size=8, shuffle=True)
test_loader = DataLoader(test_tokens, batch_size=8)
 
# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                             num_training_steps=len(train_loader) * 3)
 
# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()
 
for epoch in range(3):
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
 
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
 
# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == batch["label"]).sum().item()
        total += batch["label"].size(0)
 
print(f"Test Accuracy: {100 * correct / total:.2f}%")