import os
import json
import time
from multiprocessing.managers import BaseManager
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

#initiatlize manager connection
# TODO: change password handling
manager = BaseManager(('127.0.0.1', 5602), b'password')
manager.register('query_index')
manager.register('chat_index')
manager.register('insert_into_index')
manager.register('get_documents_list')
manager.connect()

@app.route("/chat", methods=["POST"])
def chat():
    global manager
    requestData = request.get_json()
    print(requestData)
    systemPrompt = requestData["systemPrompt"]
    print(systemPrompt)

    response = manager.chat_index(requestData)
    
    return make_response(jsonify(response)), 200


@app.route("/query", methods=["GET"])
def query_index():
    global manager
    query_text = request.args.get("text", None)
    
    data = json.loads(query_text)["content"]
    if query_text is None:
        return "No query found, please include a ?text=something parameter in the URL", 400
    
    if data.startswith("MSFT Sustainability Manager"):
        time.sleep(0.3)
        return "{\"text\": \"API call is queued and once it is finished the detail status will be sent back\"}", 202
    
    response = manager.query_index(data)._getvalue()
    print(response)
    response_json = {
        "text": str(response)
        # "sources": [{"text": str(x.source_text),
        #              "similarity": round(x.similarity, 2),
        #              "doc_id": str(x.doc_id),
        #              "start": x.node_info['start'],
        #              "end": x.node_info['end']
        #              } for x in response.source_nodes]                
    }
    return make_response(jsonify(response_json)), 200

@app.route("/uploadfile", methods=["POST"])
def upload_file():
    global manager
    if 'file' not in request.files:
        return "please send a POST with a file", 400
    
    filepath = None
    try:
        uploaded_file = request.files["file"]
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join('iBEA2-demo-documents', os.path.basename(filename))
        print("filepath=" + filepath)
        uploaded_file.save(filepath)

        if request.form.get("filename_as_doc_id", None) is not None:
            manager.insert_into_index(filepath, doc_id=filename)
        else:
            manager.insert_into_index(filepath)

    except Exception as e:
        if filepath is not None and os.path.exists(filepath):
            os.remove(filepath)
        return "Error: {}".format(str(e)), 500
    
    #if filepath is not None and os.path.exists(filepath):
    #    os.remove(filepath)

    return "File inserted", 200

@app.route("/getDocuments", methods=["GET"])
def get_document():
    document_list = manager.get_documents_list()._getvalue()

    return make_response(jsonify(document_list)), 200


@app.route("/")
def home():
    return "Welcome to th iBAE2 App"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5601)


    