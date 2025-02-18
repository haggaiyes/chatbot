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

st.set_page_config(page_title='autodoc',
layout='centered',
initial_sidebar_state='auto')

# load the environment variables
load_dotenv()
groq_api_key=os.getenv('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')
huggingface_api_key = os.getenv('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')

# load the llm model, in this case, we use llama3 model
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')

# create a prompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided text only.
Please provide the most accurate responses based on the question.
If answer cannot find from the context, please reply to the users that the information is not found in the provided documents.
If answer cannot be found from the context, please reply to the user that the information is not found in the provided documents.

<context>
{context}
<context>
Questions:{input}
Questions: {input}
"""
)

description = '''
this chatbot analyzes the uploaded document and is able to engage in a conversation understanding the context of the document.
This chatbot analyzes the uploaded document and is able to engage in a conversation understanding the context of the document.
'''

# function to clear the session state
@@ -50,8 +50,8 @@
# function to load data, split data into chunks, perform embeddings and store in vector database
def vector_embeddings(file):
if 'vectors' not in st.session_state:
        st.session_state.embeddings=HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':False})
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device':'cpu'}, encode_kwargs={'normalize_embeddings':False})
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
st.session_state.docs = []
st.session_state.final_documents = []

@@ -83,26 +83,43 @@
if file:
vector_embeddings(file)



# Streamli UI --- clear session state (vector DB)
# Streamlit UI --- clear session state (vector DB)
if st.sidebar.button('Refresh'):
clear_session_state()

# Streamlit UI --- user and bot conversation boxes
user = st.chat_message('User')
bot = st.chat_message('Assistant')
# Initialize chat history in session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Streamlit UI --- for user to input their queries
prompt1 = st.chat_input('Please enter your question:')

# initiate the QA retrieval and provide answer to user
try:
    user.write(f'User: {prompt1}')
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever, document_chain)
    response=retrieval_chain.invoke({'input':prompt1})
    bot.write(f'Bot: {response["answer"]}')
except:
    bot.write('Bot: I will only answer question based on the document uploaded...')
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
