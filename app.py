import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --------------------------- CONFIGURATION ---------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "models/gemini-2.5-flash"
  # or gemini-1.5-flash-latest for faster, lighter output

# --------------------------- CUSTOM PROMPT ---------------------------
CUSTOM_PROMPT = """
You are a precise legal AI assistant specializing in vehicle laws and compliance in India.
Your answers must be based ONLY on the following context from verified legal sources.

If the answer cannot be found in the provided context, reply exactly:
"I am sorry, but I cannot find that specific information in the provided legal documents."

CONTEXT:
{context}

QUESTION:
{question}

HELPFUL ANSWER:
"""
# --------------------------------------------------------------------


def get_conversational_chain(vector_store):
    """Build a retrieval-augmented generation (RAG) chain."""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        # Prompt Template
        prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT)

        # LLM
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=api_key,
            temperature=0.2,  # lower = more factual
        )

        # Chain: retrieval → prompt → LLM → output parser
        rag_chain = (
            {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
             "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain

    except Exception as e:
        st.error(f"Error initializing RAG chain: {e}")
        return None


def main():
    st.set_page_config(page_title="🚗 Vehicle Legality Chatbot", page_icon="⚖️")
    st.title("🚗 Vehicle Legality Chatbot")
    st.caption("Ask me about vehicle laws and compliance based only on uploaded legal documents.")

    # --------------------------- Load Vector Store ---------------------------
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, google_api_key=api_key
        )

        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        st.sidebar.success("✅ Legal knowledge base loaded successfully!")

    except Exception as e:
        st.sidebar.error(f"Error loading vector store: {e}")
        st.error("Could not load vector store. Please run `python ingest_data.py` first.")
        st.stop()

    # --------------------------- Initialize Chain ---------------------------
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversational_chain(vector_store)

    if not st.session_state.conversation:
        st.warning("Failed to initialize chatbot. Please check your API key and configuration.")
        st.stop()

    # --------------------------- Chat History ---------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I’m your AI legal assistant. How can I help you today?"}
        ]

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --------------------------- User Input ---------------------------
    user_question = st.chat_input("Ask your question here...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Analyzing legal context..."):
            try:
                response = st.session_state.conversation.invoke(user_question)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                with st.chat_message("assistant"):
                    st.markdown(response)

            except Exception as e:
                st.error(f"Error generating response: {e}")


if __name__ == "__main__":
    main()
