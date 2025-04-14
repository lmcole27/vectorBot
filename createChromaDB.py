from sentence_transformers import SentenceTransformer # type: ignore
import chromadb # type: ignore
from chromadb.config import Settings # type: ignore
from chromadb.utils import embedding_functions # type: ignore
from dataContext import segmentText, getContext, lookupMetaData
from pathlib import Path

# Create Chromadb client
client = chromadb.PersistentClient(path="./chroma_data/")
embed_model = 'all-MiniLM-L6-v2'
collection_name = "amPlan_context"

# Instantiate your embedding function and the ChromaDB collection
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(embed_model)

collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
    )

def createEmbeddings(segments, metadata):
    embeddings = embed_model.encode(segments)
    for i, (segment, embedding) in enumerate(zip(segments, embeddings)):
        collection.add(
            documents=[segment],
            embeddings=[embedding],
            ids=[f"doc_{i}"],
            metadatas=metadata
        )

# textExtractFolder = Path("./textExtract/")
# for text in textExtractFolder.iterdir():
#     with open(text, "r", encoding="utf-8") as f:
#         text = f.read()
#         segments = segmentText(text)
#         embeddings = model.encode(segments)

#         for i, (segment, embedding) in enumerate(zip(segments, embeddings)):
#             collection.add(
#                 documents=[segment],
#                 embeddings=[embedding],
#                 ids=[f"doc_{i}"],
#                 metadatas=metadata
#             )
#         print("created collection")
