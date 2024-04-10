import PyPDF2
import re
import langchain
import getpass
import os 
from langchain_openai import OpenAIEmbeddings

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def load_documents(file_paths):
    documents = []
    for file_path in file_paths:
        text = extract_text_from_pdf(file_path)
        text = preprocess_text(text)
        documents.append(preprocess_text(text))
    return documents

def create_vector_store(documents):
    embeddings = langchain.embeddings.OpenAIEmbeddings()
    document_embeddings = embeddings.embed_documents(documents)
    vector_store = langchain.vectorstores.FAISS.from_embeddings(document_embeddings, documents)
    return vector_store


"""
Next steps:
- Implement hybrid search
- Implement vector search with OpenAI embeddings
"""



if __name__ == '__main__':
    # file_paths = ["rai_assignment/nrma-car-pds-1023-east.pdf", "rai_assignment/POL011BA.pdf"]
    # documents = load_documents(file_paths)
    # vector_store = create_vector_store(documents)
    # print(documents)

    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    text = "This is a test document."
    query_result = embeddings.embed_query(text)
    query_result[:5]

