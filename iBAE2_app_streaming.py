import os
import json
import time

import pickle
from dotenv import load_dotenv
load_dotenv()

#NOTE: set up the OPENAI_API_KEY in the OS environment
OPENAI_API_KEY = os.environ['OPENAI_API_KEY'] 

from flask import Flask, request, Response, jsonify, make_response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

# from orchestrator import orchestrator
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, Document, ServiceContext, StorageContext, load_index_from_storage
from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
from llama_index.llms import PaLM, OpenAI
from llama_index.prompts import PromptTemplate
from llama_index.llms import ChatMessage, MessageRole

app = Flask(__name__)
CORS(app)

index = None
stored_docs = {}
# lock = Lock()

index_name = ".\saved_index"
pkl_name = "stored_documents.pkl"
# PROMPT_PREFIX = "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group.Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. Last, do not show thought steps back to the users. The question is... \\n";
PROMPT_PREFIX = "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group.Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. Last, do not show thought steps back to the users. The question is... \\n";

PROMPT_POSTFIX = "\\n... For numbers try the best to put them into a table format with years as row and numbers as column. For a list of things first summarize the findings then use the markdown to list the data"


def initialize_index():
    global index, stored_docs
    llm = OpenAI(temperature=0.1, model="gpt-4")
    service_context = ServiceContext.from_defaults(chunk_size_limit=512, llm=llm)

    if os.path.exists(index_name):
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name), service_context=service_context)
        print("done indexing")
    else:
        index = GPTVectorStoreIndex([], service_context=service_context)
        index.storage_context.persist(persist_dir=index_name)
    if os.path.exists(pkl_name):
        with open(pkl_name, "rb") as f:
            stored_docs = pickle.load(f)

def get_index(query_text):
    global index
    #print("query text:" + query_text)
    query_engine = index.as_query_engine(streaming=True, response_mode='tree_summarize', verbose=True,)
    #print(response)
    response_stream = query_engine.query(query_text + PROMPT_POSTFIX)
    response_stream.print_response_stream()
    return response_stream

#initiatlize manager connection
# TODO: change password handling
# manager = BaseManager(('127.0.0.1', 5602), b'password')
# manager.register('query_index')
# manager.register('chat_index')
# manager.register('insert_into_index')
# manager.register('get_documents_list')
# manager.connect()

# @app.route("/chat", methods=["POST"])
# def chat():
#     global manager
#     requestData = request.get_json()
#     print(requestData)
#     systemPrompt = requestData["systemPrompt"]
#     print(systemPrompt)

#     response = manager.chat_index(requestData)
    
#     return make_response(jsonify(response)), 200


@app.route("/query", methods=["POST"])
# @stream_with_context
def query_index():
    global index
    postdata = request.json;
    print(postdata)
    # data = postdata["message"]["content"]
    data = postdata["content"]
    
    # data = query_text["content"]
    print("query text:" + data)
    if data is None:
        return "No query found, please include a ?text=something parameter in the URL", 400
    
    if data.startswith("MSFT Sustainability Manager"):
        time.sleep(0.3)
        return "{\"text\": \"API call is queued and once it is finished the detail status will be sent back\"}", 202
   
    # response_stream.print_response_stream()
    query_engine = index.as_query_engine(streaming=True, verbose=True,)
    response_stream = query_engine.query(data + PROMPT_POSTFIX)

    def generate():
        
        # yield "event:start\ndata: stream\n\n"
        for text in response_stream.response_gen:
            yield f"event:message\ndata: {text}\r\n\r\n"
        yield "event:end\ndata: stream_end\n\n"
    # return make_response(generate())
    # return Response(stream_with_context(generate()), status=200, content_type='application/text')
    return Response(generate(), mimetype='application/text')

# @app.route("/query_backup", methods=["GET"])
# def query_index_backup():
#     global manager
#     query_text = request.args.get("text", None)
    
#     data = json.loads(query_text)["content"]
#     if query_text is None:
#         return "No query found, please include a ?text=something parameter in the URL", 400
    
#     if data.startswith("MSFT Sustainability Manager"):
#         time.sleep(0.3)
#         return "{\"text\": \"API call is queued and once it is finished the detail status will be sent back\"}", 202
    
#     response = manager.query_index(data)._getvalue()
#     print(response)
#     response_json = {
#         "text": str(response)
#         # "sources": [{"text": str(x.source_text),
#         #              "similarity": round(x.similarity, 2),
#         #              "doc_id": str(x.doc_id),
#         #              "start": x.node_info['start'],
#         #              "end": x.node_info['end']
#         #              } for x in response.source_nodes]                
#     }
#     return make_response(jsonify(response_json)), 200

@app.route("/bizTaskService", methods=["POST"])
def biz_service():
    requestData = request.get_json()
    systemPrompt = requestData["prompt"]
    agents = requestData["agent"]
    dataset = requestData["dataset"]
    model = requestData["model"]
    temperature = requestData["temperature"]
    messages = requestData["messages"]
    key = requestData["key"]
    print(model["id"] + "-" + key + "-" + systemPrompt + "-" + agents +"-" + dataset )
    time.sleep(3)
    # returnMsg = orchestrator.run(model, temperature, messages, systemPrompt, agents, dataset, key)
    # return "{\"status\": \"${returnMsg}\"}", 200
    return "{\"status\": \"system is ready for use - your personalized agent is at your service\"}", 200


# @app.route("/uploadfile", methods=["POST"])
# def upload_file():
#     global manager
#     if 'file' not in request.files:
#         return "please send a POST with a file", 400
    
#     filepath = None
#     try:
#         uploaded_file = request.files["file"]
#         filename = secure_filename(uploaded_file.filename)
#         filepath = os.path.join('iBEA2-demo-documents', os.path.basename(filename))
#         print("filepath=" + filepath)
#         uploaded_file.save(filepath)

#         if request.form.get("filename_as_doc_id", None) is not None:
#             manager.insert_into_index(filepath, doc_id=filename)
#         else:
#             manager.insert_into_index(filepath)

#     except Exception as e:
#         if filepath is not None and os.path.exists(filepath):
#             os.remove(filepath)
#         return "Error: {}".format(str(e)), 500
    
#     #if filepath is not None and os.path.exists(filepath):
#     #    os.remove(filepath)

#     return "File inserted", 200

# @app.route("/getDocuments", methods=["GET"])
# def get_document():
#     document_list = manager.get_documents_list()._getvalue()

#     return make_response(jsonify(document_list)), 200


@app.route("/")
def home():
    return "Welcome to th iBAE2 App"

if __name__ == "__main__":
    initialize_index()
    app.run(host="0.0.0.0", port=5601)
    


    