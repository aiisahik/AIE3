import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from llama_parse import LlamaParse

# GLOBAL SCOPE - ENTIRE APPLICATION HAS ACCESS TO VALUES SET IN THIS SCOPE #
# ---- ENV VARIABLES ---- # 
"""
This function will load our environment file (.env) if it is present.

NOTE: Make sure that .env is in your .gitignore file - it is by default, but please ensure it remains there.
"""
load_dotenv()

"""
We will load our environment variables here.
"""
HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]
LLAMA_CLOUD_API_KEY = os.environ["LLAMA_CLOUD_API_KEY"]

# ---- GLOBAL DECLARATIONS ---- #

# -- RETRIEVAL -- #
"""
1. Load Documents from Text File
2. Split Documents into Chunks
3. Load HuggingFace Embeddings (remember to use the URL we set above)
4. Index Files if they do not exist, otherwise load the vectorstore
"""
### 1. CREATE TEXT LOADER AND LOAD DOCUMENTS
### NOTE: PAY ATTENTION TO THE PATH THEY ARE IN. 

## THIS IS THE OLD WAY WITH THE STANDARD PyPDFLoader
# document_loader = PyPDFLoader("./data/airbnb10k-2024q1.pdf")
# documents = document_loader.load()


## THIS IS THE NEW WAY WITH LlamaParse markdown loader
llama_parser = LlamaParse(
    api_key=LLAMA_CLOUD_API_KEY,  # can also be set in your env as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
    parsing_instruction = """
        The provided document is a quarterly report filed by AirBNB,
        with the Securities and Exchange Commission (SEC).
        Some of the pages include detailed financial information about the company's performance for a specific quarter.
        It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
        It contains many tables.
        Try to be precise while answering the questions
    """
)

llama_parse_documents = llama_parser.load_data("./data/airbnb10k-2024q1.pdf")
documents = [doc.to_langchain_format() for doc in llama_parse_documents]

### 2. CREATE TEXT SPLITTER AND SPLIT DOCUMENTS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
split_documents = text_splitter.split_documents(documents)

### 3. LOAD HUGGINGFACE EMBEDDINGS
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=HF_EMBED_ENDPOINT,
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN,
)
vectorstore = None
if os.path.exists("./vectorstore"):
    num_files = len([entry for entry in os.listdir("./vectorstore") if os.path.isfile(os.path.join("./vectorstore", entry))])
    if num_files > 0:
        vectorstore = FAISS.load_local(
            "./vectorstore", 
            hf_embeddings, 
            allow_dangerous_deserialization=True # this is necessary to load the vectorstore from disk as it's stored as a `.pkl` file.
        )
        hf_retriever = vectorstore.as_retriever()
        print("Loaded Vectorstore")
if vectorstore is None:
    print("Indexing Files")
    os.makedirs("./vectorstore", exist_ok=True)
    ### 4. INDEX FILES
    ### NOTE: REMEMBER TO BATCH THE DOCUMENTS WITH MAXIMUM BATCH SIZE = 32
    for i in range(0, len(split_documents), 32):
        if i == 0:
            vectorstore = FAISS.from_documents(split_documents[i:i+32], hf_embeddings)
            continue
        print('adding documents to vector store', i, i+32, len(split_documents))
        vectorstore.add_documents(split_documents[i:i+32])

hf_retriever = vectorstore.as_retriever()

# -- AUGMENTED -- #
"""
1. Define a String Template
2. Create a Prompt Template from the String Template
"""
### 1. DEFINE STRING TEMPLATE
RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

### 2. CREATE PROMPT TEMPLATE
rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# -- GENERATION -- #
"""
1. Create a HuggingFaceEndpoint for the LLM
"""
### 1. CREATE HUGGINGFACE ENDPOINT FOR LLM
hf_llm = HuggingFaceEndpoint(
    endpoint_url=HF_LLM_ENDPOINT,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
    huggingfacehub_api_token=HF_TOKEN
)

@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message. 

    In this case, we're overriding the 'Assistant' author to be 'Paul Graham Essay Bot'.
    """
    rename_dict = {
        "Assistant" : "AirBnB Analyst Bot"
    }
    return rename_dict.get(original_author, original_author)

@cl.on_chat_start
async def start_chat():
    """
    This function will be called at the start of every user session. 

    We will build our LCEL RAG chain here, and store it in the user session. 

    The user session is a dictionary that is unique to each user session, and is stored in the memory of the server.
    """

    ### BUILD LCEL RAG CHAIN THAT ONLY RETURNS TEXT
    lcel_rag_chain = lcel_rag_chain = {
        "context": itemgetter("query") | hf_retriever, "query": itemgetter("query")
    } | rag_prompt | hf_llm

    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    async for chunk in lcel_rag_chain.astream(
        {"query": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()