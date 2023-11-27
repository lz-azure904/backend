import logging
import sys
import os
import pickle
import json

from multiprocessing.managers import BaseManager

from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import (AgentExecutor, Tool, ZeroShotAgent, initialize_agent, load_tools)
from langchain import LLMChain, OpenAI

# from llama_index.llms import PaLM, OpenAI
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, Document, ServiceContext, StorageContext, load_index_from_storage
from llama_index import VectorStoreIndex, SimpleDirectoryReader

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

#initiatlize manager connection
# TODO: change password handling
manager = BaseManager(('127.0.0.1', 5602), b'password')
manager.register('query_index')
manager.register('chat_index')
manager.register('insert_into_index')
manager.register('get_documents_list')
manager.connect()

index_name = ".\saved_index"
pkl_name = "stored_documents.pkl"

global index, stored_docs

def getLlamaIndex(llm):
    service_context = ServiceContext.from_defaults(chunk_size=512, llm=llm)
    if os.path.exists(index_name):
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name), service_context=service_context)
    else:
        index = GPTVectorStoreIndex([], service_context=service_context)
        index.storage_context.persist(persist_dir=index_name)
    if os.path.exists(pkl_name):
        with open(pkl_name, "rb") as f:
            stored_docs = pickle.load(f)
    return index

#input - LlamaIndex RAG:Redis Y,Fine Tuning Agent:GPT-4 GPT-4-UDA-01,
#implementation - can we use LLM to discover the proper tools based on the value passed 
def setupTools(tools):


    return load_tools("")

def run(model, temperature, userMsg, systemPrompt, tools, datasources, key):
    global manager
    llm = OpenAI(openai_api_key=key, temperature=temperature, model=model)
    tools = setupTools(tools)
    index = getLlamaIndex(llm)

    llama_index_data = Tool(
        name="LlamaIndex",
    #     func=lambda q: str(index.as_query_engine().query(q)),
        func=lambda q: str(index.as_query_engine(response_mode='tree_summarize', verbose=True,).query(q)),
        description="useful for when you want to answer any questions. The input to this tool should be a complete english sentence.",
        return_direct=True,
    )

    # Add python_repl to our list of tools
    tools = load_tools(["python_repl"])

    # Define our voter_data tool

    # Set a description to help the LLM know when and how to use it.
    description = (
        "Useful for when you need to answer questions about voters. "
        "You must not input SQL. Use this more than the Python tool if the question "
        "is about voter data, like 'how many DEM voters are there?' or 'count the number of precincts'"
    )


    tools.append(llama_index_data)

    # Standard prefix
    prefix = "Fulfill the following request as best you can. You have access to the following tools:"

    # Remind the agent of the Data tool, and what types of input it expects
    suffix = (
        "Pass the relevant portion of the request directly to the Data tool in its entirety."
        "\n\n"
        "Request: {input}\n"
        "{agent_scratchpad}"
    )

    # The agent's prompt is built with the list of tools, prefix, suffix, and input variables
    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix, input_variables=["input", "agent_scratchpad"]
    )

    # Set up the llm_chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Specify the tools the agent may use
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)

    # Create the AgentExecutor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )

    # set Logging to DEBUG for more detailed outputs
    memory = ConversationBufferMemory(memory_key="chat_history")
    llm = ChatOpenAI(temperature=0)
    agent_executor = initialize_agent(
        tools, llm, agent="conversational-react-description", memory=memory
    )


    request = "Show a bar graph visualizing the answer to the following question:" \
            "what is the bp aim 3 net zero sales trends for the last few years?"

    return agent_executor.run(request)