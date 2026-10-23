import importlib
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables (e.g. OPENROUTER_API_KEY)
load_dotenv()

# Step 1: Import variables from 2_retrieval_pipeline.py
# In Python, modules starting with numbers (like '2_retrieval_pipeline') cannot be imported
# using standard 'import 2_retrieval_pipeline' syntax. We use importlib.import_module() instead.
retrieval = importlib.import_module("2_retrieval_pipeline")

query = retrieval.query
relevant_docs = retrieval.relevant_docs

# Step 2: Format context and prompt for the LLM
combined_input = f"""
Answer the question using ONLY the documents below.
- Do not add facts that are not written in the documents.
- If the documents give dates or events, report them exactly as stated.
- If the documents don't contain the answer, say "I don't know".

Question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}
"""

# Step 3: Initialize the Chat Model (via OpenRouter)
model = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",   # Model ID from OpenRouter
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# Step 4: Construct system and user messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input)
]

# Step 5: Invoke the model and display the answer
print("Generating answer with LLM...")
result = model.invoke(messages)

print("\n===== Generated Message =====")
print(result.content)
