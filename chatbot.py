#set python environment
#import packages
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 
from langchain_community.llms import Replicate
from dotenv import load_dotenv

#load api keys from env file
load_dotenv()

#access the api keys
replicate_api_key=os.getenv("REPLICATE_API_TOKEN")
huggingfacehub_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
pinecone_api_key=os.getenv("PINECONE_API_KEY")

# Extract the data from the pdf book
def load_data(data):
    loader = DirectoryLoader(data,
    glob="*.pdf",
    loader_cls=PyPDFLoader)

    documents = loader.load()
    return documents
#access the data
extracted_pdf = load_data("data/")

#create text Chunks
def text_split(extracted_pdf):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                   chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_pdf)
    return text_chunks 
text_chunks = text_split(extracted_pdf)

#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embeddings
embeddings = download_hugging_face_embeddings()
embeddings

query_result = embeddings.embed_query("Hello World!")
print("Length:", len(query_result))
query_result


index_name="project"
pc=Pinecone(api_key=pinecone_api_key)
index=pc.Index("project")

#Creating Embeddings for Each of The Text Chunks & storing
docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

query = "which toyota model?"
docs = docsearch.similarity_search(query=query, k=3)

prompt_template = """
Use the following piece of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to generate any random answer from your own

Context:{context}
Question:{question}

Only return the helpful answer and nothing else
helpful answer:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context","question"])
chain_type_kwargs = {"prompt":PROMPT}

llm=Replicate(model="meta/llama-2-7b-chat")

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

while True:
    user_input = input(f"Input Prompt: ")
    result = qa({"query":user_input})
    print("Response:", result["result"])
