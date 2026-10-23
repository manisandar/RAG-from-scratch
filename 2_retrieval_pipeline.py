from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from dotenv import load_dotenv

# load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs = {"normalize_embeddings": False}
    )

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "CEO of Google?"

retriever = db.as_retriever(search_kwargs={"k":3})

# retriever = db.as_retriever(
#     search_type = "similarity_score_threshold",
#     search_kwargs={
#         "k": 5,
#         "score_threshold" : 0.3
#     }
# )

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")
#result
print("-----Context-----")
for i,doc in enumerate(relevant_docs, 1):
    print(f"Document {i}: \n {doc.page_content}\n")