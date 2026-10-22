import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):

    """ Load all text files from the docs directory"""
    print(f"Loading docs from {docs_path}")

    #check if docs dir exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The dir {docs_path} is not found")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt", #focus only on .txt files
        loader_cls=TextLoader #txt files loader, if load pdf, pdf loader or for csv, use csvloader
    )    

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt files found in {docs_path}")
    
    for i,doc in enumerate(documents): #show docs data
        print(f"\nDocuments {i+1}")
        print(f"    Source: {doc.metadata["source"]}")
        print(f"    Content Length: {len(doc.page_content)} characters")
        print(f"    Context preview: {doc.page_content[:100]}...")
        print(f"    Matadata: {doc.metadata}")
    
    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("Splitting docs into chinks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]): # show 5 chunks
            print(f"\n--Chunk {i+1}----")
            print(f"Source: {chunk.metadata["source"]}")
            print(f"Length: {len(chunk.page_content)}characters")
            print(f"Content:\n{chunk.page_content}")
            print(f"*"*50)

        if len(chunks) > 5:
            print(f"\n .... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embeddings and storing in ChromaDB...")
        
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ChromaDB vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")
    
    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore

def main():
    #1. Loading the fiels
    documents = load_documents(docs_path="docs")

    #2 Chunking the files
    chunks = split_documents(documents=documents)

    #3 Embedding and storing in Vector DB
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()
