# --- imports ---
import pprint as pp
from sentence_transformers import SentenceTransformer

# --- initialize model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

print("MODEL:")
print(model)

# --- try model ---
sentences = ['This framework generates embeddings for each input sentence', 'Including this one']
embeddings = model.encode(sentences)

print("\n\n")
print("SENTENCES:")
pp.pprint(sentences)
print()
print("EMBEDDINGS:")
##pp.pprint(embeddings)
print("(commented out)")
print("dimensions of embeddings:")
print(embeddings.shape)