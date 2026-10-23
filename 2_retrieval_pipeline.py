from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import os

load_dotenv()

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


# #answer generations
# combined_input = f"""
# Answer the question using ONLY the documents below.
# - Do not add facts that are not written in the documents.
# - If the documents give dates or events, report them exactly as stated.
# - If the documents don't contain the answer, say "I don't know".

# Question: {query}

# Documents:
# {chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
# """

# model = ChatOpenAI(
#     model="nvidia/nemotron-3-ultra-550b-a55b:free",   # any model ID ending in :free
#     api_key=os.getenv("OPENROUTER_API_KEY"),
#     base_url="https://openrouter.ai/api/v1",
#     temperature=0,
# )

# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content=combined_input)
# ]

# result = model.invoke(messages)

# print(f"Generated Message")
# print(f"{result.content}")