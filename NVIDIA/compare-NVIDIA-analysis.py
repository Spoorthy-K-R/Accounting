from sentence_transformers import SentenceTransformer, util

base_path = '/Users/spoorthy/Projects/Accounting/NVIDIA/'

# Load your summaries from files
with open(base_path + 'output-txt-yahoo/NVIDIA_yahoo_analysis.txt', 'r', encoding='utf-8') as f:
    yahoo_summary = f.read()
with open(base_path + 'output-txt/NVIDIA_analysis-new.txt', 'r', encoding='utf-8') as f:
    edgar_summary = f.read()

# Load a pre-trained Sentence Transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2') #0.589
# model = SentenceTransformer('all-mpnet-base-v2') #0.622
model = SentenceTransformer('paraphrase-MiniLM-L6-v2') #0.615

# Encode both summaries into embeddings
embeddings = model.encode([yahoo_summary, edgar_summary], convert_to_tensor=True)

# Compute cosine similarity
similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

print(f"Cosine similarity between Yahoo and EDGAR summaries: {similarity.item():.3f}")