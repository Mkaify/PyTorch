from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
 
# Sample corpus
corpus = [
    "I love pizza and pasta.",
    "Burgers are my favorite food.",
    "Cats are playful pets.",
    "Dogs are loyal companions.",
    "I'm learning machine learning.",
    "Neural networks are fascinating.",
    "Sushi is a traditional Japanese dish.",
    "Transformers changed natural language processing.",
    "I enjoy deep learning research.",
    "Parrots can mimic human speech."
]
 
# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(corpus)
 
# Apply KMeans clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)
 
# Print clustered sentences
print("\nClustered Sentences:")
for i in range(num_clusters):
    print(f"\nCluster {i + 1}:")
    for j, label in enumerate(labels):
        if label == i:
            print("-", corpus[j])