from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import time
import glob
import helper

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
splits = []

load_dotenv()

def loadData():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for file in glob.glob('data/**/*.docx', recursive=True):
        loader = Docx2txtLoader(file)
        docs = loader.load()
        splits.append(text_splitter.split_documents(docs))

    for file in glob.glob('data/**/*.pdf', recursive=True):
        docs = helper.extract_pdf_text(file)
        splits.append(text_splitter.split_documents(docs))

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

loadData()

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    question = input()
    if (question == "exit"):
        break
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
