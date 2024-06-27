from typing import List
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    MetadataInfo,
    VectorStoreInfo,
)
from pydantic import BaseModel, Field

TOP_K_RESULTS = 3


def get_auto_retrieve_fn(_index):
    def _auto_retrieve_fn(query: str, filter_key_list: List[str], filter_value_list: List[str]):
        """Auto retrieval function.

        Performs auto-retrieval from a vector database, and then applies a set of filters.

        """
        query = query or "Query"

        if _index is None:
            raise ValueError("index not found")

        exact_match_filters = [ExactMatchFilter(key=k, value=v) for k, v in zip(filter_key_list, filter_value_list)]
        retriever = VectorIndexRetriever(
            _index, filters=MetadataFilters(filters=exact_match_filters), top_k=TOP_K_RESULTS
        )
        query_engine = RetrieverQueryEngine.from_args(retriever)

        response = query_engine.query(query)
        return str(response)

    return _auto_retrieve_fn


vector_store_info = VectorStoreInfo(
    content_info="semantic information about movies",
    metadata_info=[
        MetadataInfo(
            name="title",
            type="str",
            description='title of the movie, one of ["Dune (2021 film)", "Dune - Part Two", "The Lord of the Rings - The Fellowship of the Ring", "The Lord of the Rings - The Two Towers"]',
        )
    ],
)

AUTO_RETRIEVE_TOOL_DESCRIPTION = f"""\
Use this tool to look up non-review based information about films.
The vector database schema is given below:
{vector_store_info.json()}
"""


class AutoRetrieveModel(BaseModel):
    query: str = Field(..., description="natural language query string")
    filter_key_list: List[str] = Field(..., description="List of metadata filter field names")
    filter_value_list: List[str] = Field(
        ..., description=("List of metadata filter field values (corresponding to names specified in filter_key_list)")
    )


def get_auto_retrieve_tool(index):
    auto_retrieve_tool = FunctionTool.from_defaults(
        fn=get_auto_retrieve_fn(index),
        name="semantic-film-info",
        description=AUTO_RETRIEVE_TOOL_DESCRIPTION,
        fn_schema=AutoRetrieveModel,
    )
    return auto_retrieve_tool
