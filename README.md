# 🚗 Vehicle Legality Chatbot (India)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)

This is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain, and Google's Gemini models. It's designed to act as a specialized legal assistant, answering questions about Indian vehicle laws **only** based on the legal documents you provide.

The AI is strictly instructed to *only* use the information from the uploaded documents and will refuse to answer if the context is not available.

## ⚖️ Features

* **Strictly Contextual:** Answers are generated *only* from the content of the PDF documents in the `documents` folder.
* **Chat Interface:** A simple and clean chat interface powered by Streamlit.
* **Powered by Gemini:** Uses Google's `gemini-2.5-flash` for fast and factual responses and `text-embedding-004` for high-quality retrieval.
* **Local Vector Store:** Uses `FAISS` to store the document knowledge base locally in the `vector_store` folder.

---

## 🛠️ Technology Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **AI Framework:** [LangChain](https://www.langchain.com/)
* **LLM:** [Google Gemini 2.5 Flash](https://ai.google.dev/)
* **Embeddings:** [Google `text-embedding-004`](https://ai.google.dev/)
* **Vector Store:** [FAISS-CPU](https://faiss.ai/)
* **Document Loading:** `pypdf`, `unstructured`
* **Configuration:** `python-dotenv`

---

## ⚙️ How It Works

The project is split into two main parts:

1.  **`ingest_data.py` (The Librarian)**
    * This script reads all `.pdf` files from the `/documents` folder.
    * It splits the text into small, manageable chunks.
    * It uses the Google embedding model to convert these chunks into numerical vectors.
    * It saves all these vectors into a local `FAISS` database inside the `/vector_store` folder.

2.  **`app.py` (The Chatbot)**
    * This script loads the pre-built `FAISS` vector store.
    * It starts a Streamlit web server, providing a chat window.
    * When you ask a question, the app searches the vector store for the most relevant text chunks (the "context").
    * It passes your question and that specific context to the Gemini LLM with a strict prompt: "Answer the question using *only* this context."
    * The LLM generates a response, which is then displayed in the chat.

---

## 🚀 Getting Started

### 1. Prerequisites

* Python 3.9+
* A [Google AI Studio API Key](https://aistudio.google.com/app/apikey)

### 2. Installation & Setup

1.  **Clone the repository** (or download the files to a local folder).

2.  **Create a virtual environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    * Create a file named `.env` in the main project folder.
    * Add your Google API Key to this file:

    ```ini
    # .env
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```

    > **IMPORTANT:** The `.gitignore` file is already set up to ignore `.env`, so you will not accidentally commit your secret key.

### 3. Usage

You must run the `ingest_data.py` script *before* you can run the app.

1.  **Add Your Documents:**
    * Place all your legal PDF files inside the `documents/` folder.

2.  **Create the Knowledge Base:**
    * Run the ingestion script from your terminal:
    ```bash
    python ingest_data.py
    ```
    * Wait for it to finish. You will see a `vector_store` folder appear. You only need to do this once, or again if you add, remove, or change the PDFs in the `documents` folder.

3.  **Run the Chatbot App:**
    * Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
    * A new tab will open in your web browser at `http://localhost:8501`. You can now start asking questions!

---

Legal AI Chatbot/
│
├── app.py                 # 🎯 Main Streamlit application (runs the chatbot UI)
├── ingest_data.py         # 🧠 Script to process PDFs and build the FAISS vector store
├── requirements.txt       # 📦 Python dependencies
├── .env                   # 🔑 Environment variables (e.g., API keys) — not committed to Git
├── .gitignore             # 🚫 Specifies files/folders Git should ignore
│
├── documents/             # 📂 Folder for input PDFs
│   └── example.pdf        # 📝 Example document for ingestion
│
└── vector_store/          # 🗂️ Folder for generated FAISS database
    ├── index.faiss        # 💾 FAISS index file
    └── index.pkl          # 🧩 Metadata pickle file
