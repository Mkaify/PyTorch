import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
 
# Sample dataset (positive/negative reviews)
texts = [
    "I loved this movie", "This film was fantastic", "What a great experience",
    "I hated this movie", "This film was terrible", "What a boring experience"
]
labels = [1, 1, 1, 0, 0, 0]  # 1 = positive, 0 = negative
 
# Convert text to Bag-of-Words vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()  # Convert to matrix
Y = torch.tensor(labels).float().unsqueeze(1)  # Convert labels to tensor
 
X = torch.tensor(X).float()  # Convert BoW vectors to float tensor
 
# Split into train/test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
 
# Simple classifier
class SentimentBoW(nn.Module):
    def __init__(self, input_dim):
        super(SentimentBoW, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # One layer, binary output
 
    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Sigmoid for binary classification
 
model = SentimentBoW(input_dim=X.shape[1])
 
# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
 
# Training loop
for epoch in range(100):
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
 
# Test predictions
with torch.no_grad():
    preds = model(X_test).round()
    print("\nTest Predictions:\n", preds.view(-1).tolist())