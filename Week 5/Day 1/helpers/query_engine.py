import os
from llama_index.core import SQLDatabase
from llama_index.core.indices.struct_store.sql_query import NLSQLTableQueryEngine
from llama_index.core.tools.query_engine import QueryEngineTool
import pandas as pd
from sqlalchemy import create_engine


def setup_sql_engine(movie_list):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    dune1 = pd.read_csv(os.path.abspath("data/dune1.csv"))
    dune1.to_sql("Dune (2021 film)", engine)
    dune2 = pd.read_csv(os.path.abspath("data/dune2.csv"))
    dune2.to_sql("Dune: Part Two", engine)
    lotr_fotr = pd.read_csv(os.path.abspath("data/lotr_fotr.csv"))
    lotr_fotr.to_sql("The Lord of the Rings: The Fellowship of the Ring", engine)
    lotr_tt = pd.read_csv(os.path.abspath("data/lotr_tt.csv"))
    lotr_tt.to_sql("The Lord of the Rings: The Two Towers", engine)

    sql_database = SQLDatabase(engine=engine, include_tables=movie_list)
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=movie_list,
    )

    return sql_query_engine


DESCRIPTION = """\
This tool should be used to answer any and all review related inquiries by translating a natural language query into a SQL query with access to tables:
'Dune (2021 film)' - includes data on the first movie in the Dune series,
'Dune: Part Two' - includes data on the second movie in the Dune series,
'The Lord of the Rings: The Fellowship of the Ring' - includes data on the first movie in the Lord of the Ring series,
'The Lord of the Rings: The Two Towers' - includes data on the second movie in the Lord of the Ring series,
"""


def get_sql_tool(sql_query_engine):
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_query_engine,
        name="sql-query",
        description=DESCRIPTION,
    )
    return sql_tool
