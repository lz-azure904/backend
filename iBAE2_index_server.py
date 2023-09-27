import os
import pickle
import json

#NOTE: set up the OPENAI_API_KEY in the OS environment
OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] 

from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, Document, ServiceContext, StorageContext, load_index_from_storage
from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.llms import PaLM, OpenAI
from llama_index.prompts import PromptTemplate
from llama_index.llms import ChatMessage, MessageRole

index = None
stored_docs = {}
lock = Lock()

index_name = ".\saved_index"
pkl_name = "stored_documents.pkl"


def initialize_index():
    global index, stored_docs
    llm = OpenAI(temperature=0.1, model="gpt-4")
    service_context = ServiceContext.from_defaults(chunk_size_limit=512, llm=llm)
    with lock:
        if os.path.exists(index_name):
            index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name), service_context=service_context)
        else:
            index = GPTVectorStoreIndex([], service_context=service_context)
            index.storage_context.persist(persist_dir=index_name)
        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                stored_docs = pickle.load(f)

#NOTE: not in use
def chat_index(chat_text):
    global index

    inputJSON = chat_text
    print(inputJSON)
    systemPrompt = inputJSON['systemPrompt']
    chatHistory = inputJSON['chatHistory']
    # query_engine = index.as_query_engine()
    # chat_engine = CondenseQuestionChatEngine.from_defaults(
    #     query_engine=query_engine,
    #     condense_question_prompt=systemPrompt,
    #     chat_history=chatHistory,
    #     verbose=True
    # )

    chat_engine = index.as_chat_engine(chat_mode="condense_question", streaming=False)

    # chat_engine = index.as_chat_engine()
    # streaming_response = chat_engine.chat("what is it about")
    
    streaming_response = chat_engine.chat("what is it about")
    #streaming_response.print_response_stream()
    for token in streaming_response.response_gen:
        print(token, end="")
    #response = chat_engine.
    return "ok"




def query_index(query_text):
    global index
    #print("query text:" + query_text)
    response = index.as_query_engine(response_mode='tree_summarize', verbose=True,).query(query_text)
    #print(response)
    return response

def insert_into_index(doc_file_path, doc_id=None):
    global index, stored_docs
    #print("get a file path =" + doc_file_path)
    #print("doc id =" + doc_id)
    document = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()[0]
    if doc_id is not None:
        document.doc_id = doc_id
    
    with lock:
        stored_docs[document.doc_id] = document.text[0:200]

        index.insert(document)
        index.storage_context.persist(persist_dir=index_name)

        with open(pkl_name, "wb") as f:
            pickle.dump(stored_docs, f)

    return

def get_documents_list():
    global stored_doc

    documents_list = []
    for doc_id, doc_text in stored_docs.items():
        documents_list.append({"id": doc_id, "text": doc_text})

    return documents_list

if __name__ == "__main__":
    print("initializing index...")
    initialize_index()

    manager = BaseManager(('127.0.0.1', 5602), b'password')
    manager.register('query_index', query_index)
    manager.register('chat_index', chat_index)
    manager.register('insert_into_index', insert_into_index)
    manager.register('get_documents_list', get_documents_list)
    server = manager.get_server()

    print("iBAE2 index server started..")
    server.serve_forever()