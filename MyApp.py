import streamlit as st
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
import pickle
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(layout= "wide")

with st.sidebar:
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    st.subheader("Add to the Knowledge Base")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", 
            accept_multiple_files=True)
        submitted = st.form_submit_button("Upload!")
    
    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
document_embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", model_type="query")

use_existing_vector_store = st.radio("Use existing vector store", ["Yes", "No"])
vector_store_path = "vectorstore.pkl"
raw_documents = DirectoryLoader(DOCS_DIR).load()

vector_store_exists = os.path.exists(vector_store_path)
vectorstore = None

if use_existing_vector_store == "Yes" and vector_store_exists:
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    with st.sidebar:
        st.success("Existing vector store loaded successfully")
else: 
    with st.sidebar:
        if raw_documents and use_existing_vector_store == "Yes":
            with st.spinner("Splitting documents into chunks..."):
                text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=200)
                documents = text_splitter.split_documents(raw_documents)
            with st.spinner("Adding document chunks to vector database..."):
                vectorstore = FAISS.from_documents(documents, document_embedder)
            with st.spinner("Saving vector store"):
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available")

st.subheader("Chat with your AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI assistant..."), ("user", "{input}")]
)
chain = prompt_template | llm | StrOutputParser()


user_input = st.chat_input("Ask your question:")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if vectorstore is not None and use_existing_vector_store == "Yes":
            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in docs])
            augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}\n"
        else:
            augmented_user_input = f"Question: {user_input}\n"

        for response in chain.stream({"input": augmented_user_input}):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})