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

load_dotenv()

# os.environ["OPENAI_API_KEY"] = getpass.getpass()

start_time = time.time()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Load, chunk and index the contents of the blog.
loader = Docx2txtLoader("data/List of Data Processing Systems Architecture quest.docx")
docs = loader.load()

print(time.time() - start_time)
start_time = time.time()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(time.time() - start_time)
start_time = time.time()

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

print(time.time() - start_time)
start_time = time.time()

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is system analytics?"):
    print(chunk, end="", flush=True)
