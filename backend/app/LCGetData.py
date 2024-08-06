from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import find_dotenv, load_dotenv
import os
from openai import OpenAI

load_dotenv(find_dotenv())
OpenAI.api_key = os.getenv('OPENAI_API_KEY')

def get_relevant_data(user_input, number_of_chunks, chunk_size):
    # Set the path to the policyDocuments folder relative to the current file
    data_documents_path = os.path.join(os.path.dirname(__file__), 'data')

    # Load all the PDF documents from the policyDocuments folder
    loader = DirectoryLoader(data_documents_path, glob='**/*.txt')
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create a vector store from the chunks
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Perform a similarity search against the user's input
    docs = vectorstore.similarity_search(user_input, k=number_of_chunks)

    # Combine the relevant chunks into a single string
    relevant_chunks = ' '.join([doc.page_content for doc in docs])

    return relevant_chunks