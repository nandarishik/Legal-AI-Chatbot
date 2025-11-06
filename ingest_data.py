import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from .env file
# langchain-google-genai will automatically find and use the GOOGLE_API_KEY
load_dotenv()

# Define paths
DOCUMENTS_PATH = "documents"
VECTOR_STORE_PATH = "vector_store"

def main():
    print("Starting data ingestion process...")

    # 1. Load documents
    # Use DirectoryLoader to load all PDFs from the 'documents' folder
    loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.pdf")
    try:
        documents = loader.load()
        if not documents:
            print(f"No PDF documents found in '{DOCUMENTS_PATH}'. Please add some files.")
            return
        print(f"Loaded {len(documents)} document(s).")
    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks.")

    # 3. Create embeddings model
    # This will convert our text chunks into numerical vectors
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    print("Embedding model loaded.")

    # 4. Create FAISS vector store from chunks
    print("Creating vector store from document chunks...")
    try:
        vector_store = FAISS.from_documents(docs, embeddings)
        
        # 5. Save the vector store locally
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"Vector store created and saved successfully to '{VECTOR_STORE_PATH}'.")
        print("\nData ingestion complete. You can now run the main app.")
        
    except Exception as e:
        print(f"An error occurred while creating or saving the vector store: {e}")

if __name__ == "__main__":
    main()