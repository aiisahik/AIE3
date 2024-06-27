import logging
import sys
import chainlit as cl
from dotenv import load_dotenv
from helpers.auto_retrieve import get_auto_retrieve_tool
from helpers.index import create_index
from helpers.query_engine import get_sql_tool, setup_sql_engine
import llama_index
from llama_index.core import Settings, set_global_handler
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent.runner.base import AgentRunner
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import wandb

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# ---- ENV VARIABLES ---- #
load_dotenv()

# ---- GLOBAL DECLARATIONS ---- #


set_global_handler("wandb", run_args={"project": "aiew5d1"})
wandb_callback = llama_index.core.global_handler
movie_list = [
    "Dune (2021 film)",
    "Dune: Part Two",
    "The Lord of the Rings: The Fellowship of the Ring",
    "The Lord of the Rings: The Two Towers",
]

Settings.llm = OpenAI(model="gpt-4o")
Settings.embedding = OpenAIEmbedding(model="text-embedding-3-small")

print("creating index")
index = create_index(movie_list, wandb_callback)
print("setting up sql engine")
sql_query_engine = setup_sql_engine(movie_list)

print("getting auto_retrieve_tool")
auto_retrieve_tool = get_auto_retrieve_tool(index)
print("getting sql_tool")
sql_tool = get_sql_tool(sql_query_engine)


@cl.author_rename
def rename(original_author: str):
    """
    This function can be used to rename the 'author' of a message.
    """
    rename_dict = {"Assistant": "Critic Bot"}
    return rename_dict.get(original_author, original_author)


@cl.on_chat_start
async def start():
    welcome_msg = cl.Message(
        author="Assistant",
        content="Setting things up. Please wait ...",
    )
    await welcome_msg.send()
    await cl.sleep(1)

    wandb.log({"setup_step:": "setting up agent"})
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=[auto_retrieve_tool, sql_tool],
        verbose=True,
    )

    agent = agent_worker.as_agent()
    cl.user_session.set("agent", agent)

    welcome_msg.content = "Alright, I'm ready for any questions relating to the Dune and LOTR Trilogy movies."
    await welcome_msg.update()


@cl.step(type="tool")
async def step_agent_task(source):
    return f"According to tool {source.tool_name}, {source.content}\n"


@cl.on_message
async def main(message: cl.Message):
    """
    This function will be called every time a message is recieved from a session.

    """
    agent: AgentRunner = cl.user_session.get("agent")

    final_msg = cl.Message(author="Assistant", content="Thinking hard about this ... \n")
    await final_msg.send()
    await cl.sleep(0.01)

    task = agent.create_task(message.content)
    is_last_step = False
    while not is_last_step:
        step_output = agent.run_step(task.task_id)
        if step_output.is_last:
            response = agent.finalize_response(task.task_id)
            final_msg.content = final_msg.content + f"\n\nFinal Answer: \n{str(response)}"
            await final_msg.update()
            await cl.sleep(0.01)
            is_last_step = True
        else:
            for source in step_output.output.sources:
                final_msg.content = final_msg.content + "\n" + await step_agent_task(source)
                await final_msg.update()
                await cl.sleep(0.01)
