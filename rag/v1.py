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
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain.output_parsers import SimpleStringOutputParser
from pydantic import BaseModel,Field

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate

from ragMainSetup import retriever

load_dotenv()

# =========


loader=PyPDFLoader("C:/Users/rohit/OneDrive/Desktop/personal/rp/now/opd-ai/gemini/elsummer-hackathon/rag/HowtoSwingTrade-AudioBook.pdf")
data=loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv('gemini_api'))


# Test embedding a query
vector = embeddings.embed_query("hello, world!")
print(len(vector))
print(vector[0])


vectorstoredb = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstoredb.as_retriever(search_type="similarity", search_kwargs={"k": 5})



# ====

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3,google_api_key=os.getenv('gemini_api'))

system_prompt = (
   '''You are a expert teacher who is able to find important aspects of a document which are useful for memorizing.
    Given the following documents, create a list of flashcards that are easy to memorize and capture the essence of the document.
    
    OUTPUT FORMAT: 

    Flashcard = 'front_text': str, 'back_text': str
    Return: list[Flashcard]

    '''
    "DOCUMENT: {context}"
)

# Set up the prompt for the QA chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, chain)

# Define the output template
# output_template = """
# **Strategies for Swing Trading:**
# 1. {}
# 2. {}
# """


response = rag_chain.invoke({"input": "How to always succeed in swing trading. List 2 effective strategies for same. form answer in 2 lines"})


# response_schemas = [
#     ResponseSchema(name="answer", description="answer to the user's question"),
#     ResponseSchema(
#         name="source",
#         description="source used to answer the user's question, should be a given pdf.",
#     ),
# ]

# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# format_instructions = output_parser.get_format_instructions()
# prompt = PromptTemplate(
#     template="answer the users question as best as possible.\n{format_instructions}\n{question}",
#     input_variables=["question"],
#     partial_variables={"format_instructions": format_instructions},
# )

# chain = prompt | llm | output_parser
# chain.invoke({"explain bullish candle sticks from the document using image number and print image number."})
# # Parse the output using the parser
# for s in chain.stream({"question": "what's the capital of france?"}):
#     print(s)
# print("--------",chain)


# FOMRATIING OUTPUT 

class Joke(BaseModel):
    question: str = Field(description="question from document")
    answer: str = Field(description="answer to the question asked")


# And a query intented to prompt a language model to populate the data structure.
joke_query = "explain bullish candle sticks from the document using image number and print image number."

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain= prompt | llm | parser
print(chain.invoke({"query": joke_query}))

# def format_docs(docs):
#     print("DOCS==",docs)
#     return "\n\n".join([d.page_content for d in docs])


# chain =(
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    
#     | prompt | llm | parser
# ) 

# print(chain.invoke({"query": joke_query}))
# print(response)


# chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, chain)
# response = rag_chain.invoke({"input": "How to always succeed in swing trading. List 2 effective strategies for same. form answer in 2 lines"})
