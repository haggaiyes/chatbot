# import essential libraries
import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile

# Streamlit configuration
st.set_page_config(page_title='autodoc',
                   layout='centered',
                   initial_sidebar_state='auto')

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('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')
huggingface_api_key = os.getenv('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')

# Load multiple LLM models for comparison
llm_models = {
    'Llama3-8b-8192': ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192'),
    'Llama2-7b': ChatGroq(groq_api_key=groq_api_key, model_name='Llama2-7b')  # Add more as needed
}

# Embedding models for comparison
embedding_models = {
    'BAAI/bge-small-en-v1.5': HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}),
    'sentence-transformers/all-MiniLM-L6-v2': HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
}

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided text only.
Please provide the most accurate responses based on the question.
If answer cannot be found from the context, please reply to the user that the information is not found in the provided documents.

<context>
{context}
<context>
Questions: {input}
"""
)

description = '''
This chatbot analyzes the uploaded document and engages in a conversation, understanding the context of the document.
'''

# Function to clear the session state
def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]

# Function to load data, split data into chunks, perform embeddings, and store in vector database
def vector_embeddings(file, embedding_model):
    st.session_state.embeddings = embedding_model  # use selected embedding model
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    st.session_state.docs = []
    st.session_state.final_documents = []

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        final_documents = st.session_state.text_splitter.split_documents(docs)

        # Append the new documents to the existing ones
        st.session_state.docs.extend(docs)
        st.session_state.final_documents.extend(final_documents)

        # Update the vector store with the new documents
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# UI for uploading documents
st.sidebar.title('Documents Uploader')
st.sidebar.write(description)
file = st.sidebar.file_uploader('Upload your document', accept_multiple_files=False, type=['pdf'])

# If a file is uploaded, perform vector embeddings for all embedding models
if file:
    for embedding_name, embedding_model in embedding_models.items():
        st.sidebar.write(f"Running embedding: {embedding_name}")
        vector_embeddings(file, embedding_model)

# Streamlit UI --- clear session state (vector DB)
if st.sidebar.button('Refresh'):
    clear_session_state()

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Streamlit UI --- for user to input their queries
prompt1 = st.chat_input('Please enter your question:')

if prompt1:
    # Add user's message to session state
    st.session_state.messages.append({"role": "User", "content": prompt1})

    # Display user's message
    with st.chat_message("User"):
        st.markdown(prompt1)

    # Iterate over LLM models and compare performance
    for model_name, llm in llm_models.items():
        try:
            st.sidebar.write(f"Evaluating model: {model_name}")

            # Create a document chain and retriever
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # Measure the time taken for the response
            start_time = time.time()
            response = retrieval_chain.invoke({'input': prompt1})
            end_time = time.time()

            # Calculate time elapsed
            time_taken = end_time - start_time
            bot_response = response["answer"]
            
            # Add bot's response to session state and display
            st.session_state.messages.append({"role": f"Assistant ({model_name})", "content": bot_response})
            with st.chat_message(f"Assistant ({model_name})"):
                st.markdown(f"{bot_response}\n\n_Time taken: {time_taken:.2f} seconds_")

        except Exception as e:
            st.session_state.messages.append({"role": f"Assistant ({model_name})", "content": f"Error: {e}"})
            with st.chat_message(f"Assistant ({model_name})"):
                st.markdown(f"Error: {e}")

