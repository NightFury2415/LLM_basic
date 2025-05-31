from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
import os

# 1. Load PDF
pdf_path = "data/test2.pdf"
loader = PyPDFLoader(pdf_path)
if not os.path.exists(pdf_path):
    raise FileNotFoundError("The specified PDF file does not exist.")
pages = loader.load()

# 2. Split into small chunks
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
docs = splitter.split_documents(pages)

# 3. Create vector store
embedding = HuggingFaceEmbeddings()
db = Chroma.from_documents(docs, embedding, persist_directory="db")
db.persist()

# 4. Load LLM (2048 is your model's max context window)
llm = LlamaCpp(
    model_path="model/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    temperature=0.7,
    max_tokens=256,   # <--- restrict generation size
    top_p=0.9,
    verbose=True
)

# 5. Retrieval chain — fetch fewer docs to stay under context limit
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 2}),  # Only 2 docs
    chain_type="map_reduce",                            # Safe chaining
    return_source_documents=True
)

# 6. Loop to ask questions
while True:
    query = input("Ask a question (or type 'exit' to quit): ")
    if query.lower() in ["exit", "quit"]:
        break
    try:
        result = qa.invoke({"query": query})  # <--- new invoke API
        print("\nAnswer:", result["result"])
        if result["source_documents"]:
            print("\nSources:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"  [{i}] {doc.metadata.get('source', 'N/A')}")
        else:
            print("No relevant source documents found.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
