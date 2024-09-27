import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma


load_dotenv()

loader=PyPDFLoader("./HowtoSwingTrade-AudioBook.pdf")
data=loader.load()
# print(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
docs = text_splitter.split_documents(data)

# print("Total number of Chunks: ", len(docs))  
# for chunk in docs:
#     print(chunk.page_content)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv('gemini_api'))


# Test embedding a query
vector = embeddings.embed_query("hello, world!")
print(len(vector))
print(vector[0])


vectorstoredb = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstoredb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

retrieved_docs = retriever.invoke("What is swing trading")
# print(len(retrieved_docs))
# print(retrieved_docs[0].page_content)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

system_prompt = (
   "You are a financial expert. Provide clear, concise answers based on the provided context. "
    "If the information is not found in the context, state that the answer is unavailable. "
    "Use a maximum of three sentences."
    "\n\n"
    "{context}"
)

# Set up the prompt for the QA chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Create the RAG chain
chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, chain)
response = rag_chain.invoke({"input": "How to always succeed in swing trading. List 2 effective strategies for same. form answer in 2 lines"})
print(response)