# import essential libraries
import streamlit as st
import os
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

st.set_page_config(page_title='autodoc', layout='centered', initial_sidebar_state='auto')

# load the environment variables
load_dotenv()
groq_api_key = os.getenv('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')
huggingface_api_key = os.getenv('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')

# load the llm model, in this case, we use llama3 model
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')

# create a prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided text only.
    Please provide the most accurate responses based on the question.
    If the answer cannot be found from the context, please reply to the user that you are unable to answer to that question as it is not related to the document.

    <context>
    {context}
    <context>
    Questions: {input}
    """
)

description = '''
This chatbot analyzes the uploaded document and is able to engage in a conversation understanding the context of the document.
'''

# function to clear the session state
def clear_session_state():
    for key in st.session_state.keys():
        del st.session_state[key]

# function to load data, split data into chunks, perform embeddings and store in vector database
def vector_embeddings(file):
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': False})
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

st.title('Autodoc')

st.sidebar.title('Documents Uploader')
st.sidebar.write(description)
file = st.sidebar.file_uploader('Upload your document', accept_multiple_files=False, type=['pdf'])
if file:
    vector_embeddings(file)

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

    # initiate the QA retrieval and provide answer to user
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({'input': prompt1})
        bot_response = response["answer"]
    except:
        bot_response = 'I will only answer questions based on the document uploaded...'
    
    # Add bot's response to session state
    st.session_state.messages.append({"role": "Assistant", "content": bot_response})

    # Display bot's response
    with st.chat_message("Assistant"):
        st.markdown(bot_response)
