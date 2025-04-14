import PyPDF2
import os
import uuid
from sentence_transformers import SentenceTransformer # type: ignore
import chromadb # type: ignore
from chromadb.config import Settings # type: ignore
from chromadb.utils import embedding_functions # type: ignore
import pandas as pd # type: ignore
import json
from pathlib import Path
from rapidfuzz import process, fuzz # type: ignore
#from createChromaDB import createEmbeddings


# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
chromaClient = chromadb.PersistentClient(path="./chroma_data/")
try:
    collection = chromaClient.get_collection(name="amPlan_context")
    print(f"Collection 'amPlan_context' found with {collection.count()} documents")
except Exception as e:
    print(f"Error accessing collection: {e}")
    print("Creating new collection...")
    collection = chromaClient.create_collection(
        name="amPlan_context",
        metadata={"hnsw:space": "cosine"}
    )



def lookupMetaData(input_name, threshold=50):
    # Load the metadata CSV file
    df = pd.read_csv('./sourceData/Municipality_MetaData.csv')
    name_list = df['Name'].dropna().tolist()

    # Find best match using fuzzy logic
    best_match, score, index = process.extractOne(
        input_name, name_list, scorer=fuzz.token_sort_ratio
    )
    if score >= threshold:
        matched_row = df[df['Name'] == best_match].iloc[0]
        output = matched_row.to_dict()
        print("output=", output)
        # return json.dumps(output)
        return output
    else:
        return f"No good match found for '{input_name}' (best guess: '{best_match}', score: {score})"


# Segment text
def segmentText(text, chunk_size=500, overlap=50):
    segments = []
    for i in range(0, len(text), chunk_size - overlap):
        segments.append(text[i:i + chunk_size])
    return segments


# # # Convert PDFs to text
# os.makedirs("./textExtract", exist_ok=True)    
# pdffolder = Path("./sourcePDF/")

# for pdf in pdffolder.iterdir():
#     name = str(pdf)
#     name = name[10:]
#     year = name[:4]
#     municipality = name[5:-4]
#     metadata = lookupMetaData(municipality)
#     docData = {"year":year, "municipality":municipality, "metadata":metadata}
#     print(docData)
#     with open(pdf, 'rb') as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text()  
#         # random_number = uuid.uuid4()
#         # file_path = f"./textExtract/file_{random_number}.txt"
#         # with open(file_path, "w") as f:
#         #     f.write(text)
#         segments = segmentText(text)
#         embeddings = model.encode(segments)

#         for i, (segment, embedding) in enumerate(zip(segments, embeddings)):
#             collection.add(
#                 documents=[segment],
#                 embeddings=[embedding],
#                 ids=[f"doc_{i}"],
#                 metadatas=metadata
#             )




# Function to encode user question
def getContext(model, collection, query):
    print(f"Query: {query}")
    query_embedding = model.encode(query)
    print(f"Query embedding shape: {query_embedding.shape}")
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    print(f"Query results: {results}")
    if not results['documents'] or not results['documents'][0]:
        print("No documents found in collection")
        return "No relevant context found."
    relevant_segments = results['documents'][0]
    context = "\n\n".join(relevant_segments)
    print(f"Generated context: {context}")
    return context
