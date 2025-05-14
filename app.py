# Import essential libraries
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader  # <-- Corrected loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Streamlit Page Config
st.set_page_config(page_title='Customer Support Chatbot', layout='centered', initial_sidebar_state='auto')

# Load environment variables from .env
load_dotenv()
groq_api_key = os.getenv('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') 
huggingface_api_key = os.getenv('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')  

# Load the LLM model (LLaMa3)
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')

# Prompt Template for Customer Support
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided support document only.
Please provide clear and helpful customer support responses.
If the information is not found in the document, kindly inform the user.

<context>
{context}
<context>
Question: {input}
"""
)

# Clear session state function
def clear_session_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

# Load Customer Support Data and Create Vector Store
@st.cache_resource
def load_customer_support_data():
    loader = TextLoader('data/customer_support_faq.txt')  # Static document path
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5',
                                       model_kwargs={'device': 'cpu'},
                                       encode_kwargs={'normalize_embeddings': False})
    vectors = FAISS.from_documents(final_documents, embeddings)

    return vectors

# Load vectors and store in session state
if 'vectors' not in st.session_state:
    st.session_state.vectors = load_customer_support_data()

# Streamlit App Title
st.title('Customer Support Chatbot')

# Sidebar - Refresh button to clear chat history
if st.sidebar.button('Refresh Chat'):
    clear_session_state()

# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Streamlit Chat Input
user_query = st.chat_input('Please enter your question:')

if user_query:
    # Add user's message to session state
    st.session_state.messages.append({"role": "User", "content": user_query})

    # Display user's message
    with st.chat_message("User"):
        st.markdown(user_query)

    # Run Retrieval Chain for answering
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': user_query})
        bot_response = response["answer"]
    except Exception as e:
        bot_response = f"Sorry, I encountered an error: {e}"

    # Add bot's response to session state
    st.session_state.messages.append({"role": "Assistant", "content": bot_response})

    # Display bot's response
    with st.chat_message("Assistant"):
        st.markdown(bot_response)
