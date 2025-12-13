from transformers import pipeline

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Input text
text = "This course provides a great introduction to neural networks and deep learning."

# Candidate labels (your custom categories)
candidate_labels = ["education", "technology", "sports", "finance", "cooking"]

# Perform classification
result = classifier(text, candidate_labels)

# Show results
print("Text:", text)
print("\nTop Prediction:")
print(f"{result['labels'][0]} (score: {result['scores'][0]:.2f})")

print("\nFull Rankings:")
for label, score in zip(result['labels'], result['scores']):
    print(f"{label}: {score:.4f}")