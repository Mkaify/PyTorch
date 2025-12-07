from sentence_transformers import SentenceTransformer, util
 
# Load pretrained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
 
# Corpus of documents to search
corpus = [
    "Machine learning is a subset of artificial intelligence.",
    "PyTorch is an open-source deep learning framework.",
    "Transformers have revolutionized NLP.",
    "Cats are wonderful companions.",
    "The Eiffel Tower is located in Paris."
]
 
# Encode the corpus
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
 
# Query input
query = "Where is the Eiffel Tower?"
query_embedding = model.encode(query, convert_to_tensor=True)
 
# Compute cosine similarity
cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
top_result = torch.topk(cos_scores, k=1)
 
# Display best match
print("Query:", query)
print("Most Relevant Document:", corpus[top_result.indices[0].item()])
print("Score:", top_result.values[0].item())