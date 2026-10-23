import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables from .env file (e.g., OPENAI_API_KEY)
load_dotenv()


def load_documents(docs_path="docs"):
    """
    Step 1: Load Raw Documents
    -------------------------
    Reads text files from the specified directory and loads them into LangChain Document
    objects, capturing both text content (`page_content`) and file origin (`metadata`).
    
    Args:
        docs_path (str): Relative or absolute path to the folder containing source documents.
        
    Returns:
        list[Document]: Loaded documents containing text and metadata.
    """
    print(f"Loading docs from {docs_path}")

    # Validate that the source directory exists before attempting to read
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory '{docs_path}' was not found.")

    # DirectoryLoader recursively scans the directory matching files with the glob pattern.
    # TextLoader reads plain text (.txt); for other formats, use PyPDFLoader, CSVLoader, etc.
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",           # Target only text files
        loader_cls=TextLoader   # Document loader class for reading individual files
    )    

    documents = loader.load()

    # Ensure documents were successfully discovered and loaded
    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in '{docs_path}'.")
    
    # Display summary statistics and previews for each loaded document
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}")
        print(f"    Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"    Content Length: {len(doc.page_content)} characters")
        print(f"    Context preview: {doc.page_content[:100]}...")
        print(f"    Metadata: {doc.metadata}")
    
    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """
    Step 2: Chunk Documents
    -----------------------
    Splits large documents into smaller, manageable chunks.
    Smaller chunks improve embedding precision and prevent exceeding LLM context windows.
    
    Args:
        documents (list[Document]): The loaded raw documents.
        chunk_size (int): Maximum character length per chunk.
        chunk_overlap (int): Number of overlapping characters between adjacent chunks 
                             to maintain continuity and prevent cutting context mid-thought.
                             
    Returns:
        list[Document]: List of chunked Document objects with updated metadata.
    """
    print("Splitting docs into chunks...")

    # CharacterTextSplitter splits text on a specific delimiter (defaults to double newline "\n\n")
    # while adhering as closely as possible to the chunk_size limit.
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # Preview the first few chunks to inspect chunking quality and boundary splits
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n-- Chunk {i+1} --")
            print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:\n{chunk.page_content}")
            print("*" * 50)

        if len(chunks) > 5:
            print(f"\n.... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """
    Step 3: Embeddings & Vector Storage
    -----------------------------------
    Converts each text chunk into a high-dimensional vector representation (embedding)
    using OpenAI, then stores and indexes them in a persistent ChromaDB vector database.
    
    Args:
        chunks (list[Document]): Chunked text documents to embed.
        persist_directory (str): Local disk path where ChromaDB database files will be saved.
        
    Returns:
        Chroma: Persistent Chroma vector store ready for similarity search and retrieval.
    """
    print("Creating embeddings and storing in ChromaDB...")
        
    # Initialize the embedding model.
    # text-embedding-3-small outputs 1536-dimensional vectors with high quality and low latency/cost.
    embedding_model = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs = {"normalize_embeddings": False}
    )
    
    # Chroma.from_documents takes chunks, computes their embeddings via embedding_model,
    # builds an HNSW (Hierarchical Navigable Small World) index, and persists it to disk.
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity for vector distance
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    """
    Orchestrate the Complete Ingestion Pipeline
    -------------------------------------------
    1. Load: Extract documents from disk.
    2. Split: Divide text into semantic chunks.
    3. Embed & Store: Convert chunks into vectors and save into ChromaDB.
    """
    # 1. Loading the files from docs/ directory
    documents = load_documents(docs_path="docs")

    # 2. Chunking the files into smaller segments
    chunks = split_documents(documents=documents)

    # 3. Embedding chunks and persisting to ChromaDB vector store
    vectorstore = create_vector_store(chunks)


if __name__ == "__main__":
    main()
