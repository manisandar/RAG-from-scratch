import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Load environment variables (such as OPENROUTER_API_KEY)
load_dotenv()

# ==========================================
# 1. Configuration & Model Initialization
# ==========================================

PERSIST_DIRECTORY = "db/chroma_db"

# Local embedding model (all-MiniLM-L6-v2 produces 384-dimensional dense vectors)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": False}
)

# Connect to the existing ChromaDB vector database
db = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Initialize retriever (fetches top 3 most relevant chunks)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Chat LLM via OpenRouter (handles query rewriting and answer generation)
model = ChatOpenAI(
    model="nvidia/nemotron-3-ultra-550b-a55b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# In-memory conversational memory storing alternating HumanMessage & AIMessage
chat_history = []


# ==========================================
# 2. Pipeline Functions
# ==========================================

def contextualize_question(user_question: str) -> str:
    """
    Step A: Query Reformulation (History-Aware Search)
    --------------------------------------------------
    If chat history exists, pronouns (e.g. 'he', 'they', 'it') make questions ambiguous
    for semantic search (e.g., 'Where was he born?').
    This function instructs the LLM to rewrite the question into a standalone,
    search-friendly query before querying the vector store.
    """
    if not chat_history:
        return user_question

    rewrite_messages = [
        SystemMessage(
            content=(
                "Given the chat history and the latest user question which might reference context in the history, "
                "rewrite the question to be a standalone search query. Do NOT answer the question, only return the rewritten query."
            )
        )
    ] + chat_history + [
        HumanMessage(content=f"New Question: {user_question}")
    ]

    response = model.invoke(rewrite_messages)
    rewritten_query = response.content.strip()
    print(f"[Query Rewriter] Search query: '{rewritten_query}'")
    return rewritten_query


def generate_answer(user_question: str, relevant_docs: list) -> str:
    """
    Step B: Grounded Answer Generation
    ----------------------------------
    Formats retrieved documents into a context prompt with strict anti-hallucination
    instructions, and invokes the LLM along with conversation history.
    """
    context_text = "\n".join([f"- {doc.page_content.strip()}" for doc in relevant_docs])

    system_instruction = (
        "You are a helpful assistant that answers questions based ONLY on the provided documents "
        "and conversation history.\n"
        "- Do not extrapolate or add facts not present in the documents.\n"
        "- If the documents do not contain the answer, say 'I don't know'."
    )

    user_prompt = f"""Documents:
{context_text}

Question: {user_question}"""

    generation_messages = [
        SystemMessage(content=system_instruction)
    ] + chat_history + [
        HumanMessage(content=user_prompt)
    ]

    response = model.invoke(generation_messages)
    return response.content


def ask_question(user_question: str) -> str:
    """
    Main Orchestrator
    -----------------
    1. Reformulate question with history if needed.
    2. Retrieve relevant chunks from ChromaDB.
    3. Generate grounded answer.
    4. Save user question & AI answer to conversation history.
    """
    # 1. History-aware query rewrite
    search_query = contextualize_question(user_question)

    # 2. Retrieve top-k documents
    relevant_docs = retriever.invoke(search_query)
    print(f"[Retriever] Retrieved {len(relevant_docs)} document chunk(s):")
    for i, doc in enumerate(relevant_docs, 1):
        first_line = doc.page_content.strip().split("\n")[0]
        print(f"  [{i}] {first_line[:80]}...")

    # 3. Generate answer
    print("\n[LLM] Generating answer...")
    answer = generate_answer(user_question, relevant_docs)

    # 4. Update conversation memory
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"\nAnswer:\n{answer}\n")
    return answer


# ==========================================
# 3. Interactive CLI
# ==========================================

def start_chat():
    """Starts the interactive terminal chat loop."""
    print("=" * 60)
    print("RAG Assistant with Conversational Memory")
    print("Type your question below. Type 'q', 'quit', or 'exit' to end.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        # Check for exit commands or blank input
        if user_input.lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        ask_question(user_input)


if __name__ == "__main__":
    start_chat()
