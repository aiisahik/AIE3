# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)
import chainlit as cl  # importing chainlit for our app
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessageChunk
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from langchain.docstore.document import Document
from dotenv import load_dotenv
import tiktoken
from chainlit.types import AskFileResponse
import PyPDF2
from io import BytesIO
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough

load_dotenv()

## HELPERS

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    length_function = tiktoken_len,
)

def process_file(file: AskFileResponse) -> list[Document]:
    file_stream = BytesIO(file.content)
    if file.type == "text/plain":
        documents = [Document(page_content=file.content, metadata={"source":file.name})]
    elif file.type == "application/pdf":
        pdf = PyPDF2.PdfReader(file_stream)
        pdf_text = ""
        documents = []
        for page_num, page in enumerate(pdf.pages):
            pdf_text += page.extract_text()
            documents.append(Document(page_content=page.extract_text(), metadata={"source":f"page {page_num}"}))
    
    docs = text_splitter.split_documents(documents)
    return docs


RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

Only answer the question with the provided context. If the question is irrelevant to the context, then just say that you do not know.
"""



@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():

    ## SETUP 

    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text or pdf file for me to analyze and provide answers to!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    
    cl.sleep(0.05)
    await cl.Message(content=f"Processing `{file.name}`...").send()
    await cl.sleep(0.05)

    documents = process_file(file)
    
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    qdrant_vectorstore = Qdrant.from_documents(
        documents,
        embedding_model,
        location=":memory:",
        collection_name="SOME FILE",
    )

    qdrant_retriever = qdrant_vectorstore.as_retriever()
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    openai_chat_model = ChatOpenAI(model="gpt-4o", streaming=True)

    chain = (
      # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
      # "question" : populated by getting the value of the "question" key
      # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
      {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
      # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
      #              by getting the value of the "context" key from the previous step
      | RunnablePassthrough.assign(context=itemgetter("context")
    )
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
    )

    await cl.Message(content=f"`{file.name}` processed. You can now ask questions!").send()

    cl.user_session.set("chain", chain)



@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    res = cl.Message(content="")

    async for chunk in chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        if isinstance(chunk, dict) and 'response' in chunk and isinstance(chunk['response'], AIMessageChunk) and hasattr(chunk['response'], 'content'): 
            await res.stream_token(chunk['response'].content)

    await res.send()
