from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models


def create_index(movie_list, wandb_callback):
    wiki_docs = WikipediaReader().load_data(pages=movie_list, auto_suggest=False)
    client = QdrantClient(location=":memory:")

    client.create_collection(
        collection_name="movie_wikis", vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
    )

    vector_store = QdrantVectorStore(client=client, collection_name="movie_wikis")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        [],
        storage_context=storage_context,
    )

    pipeline = IngestionPipeline(transformations=[TokenTextSplitter()])
    for movie, wiki_doc in zip(movie_list, wiki_docs):
        nodes = pipeline.run(documents=[wiki_doc])
        for node in nodes:
            node.metadata = {"title": movie}
        index.insert_nodes(nodes)

    wandb_callback.persist_index(index, index_name="movie-index-qdrant")

    return index
