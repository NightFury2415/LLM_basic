from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA
import os

# 1. Load PDF
loader = PyPDFLoader("../data/test2.pdf")  # Replace with your PDF file path
if not os.path.exists("../data/test2.pdf"):
    raise FileNotFoundError("The specified PDF file does not exist.")
pages = loader.load()

# 2. Split text into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

# 3. Create vector store
embedding = HuggingFaceEmbeddings()
db = Chroma.from_documents(docs, embedding, persist_directory="db")
db.persist()

# 4. Load LLM
llm = LlamaCpp(
    model_path="C:\\local-llm\\models\\mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    temperature=0.7,
    verbose=True
)

# 5. Set up Q&A chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

# 6. Ask questions
while True:
    query = input("Ask a question: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa({"query": query})
if result["source_documents"]:
    print("\nAnswer:", result["result"])
else:
    print("\nI don’t know. This info is not in the documents.")