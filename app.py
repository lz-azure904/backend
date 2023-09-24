import os
import datetime
import openai
from flask import Flask, redirect, render_template, session, request, url_for
from flask_session import Session  # https://pythonhosted.org/Flask-Session
import identity, identity.web
import requests
import app_config
import numpy as np
import pandas as pd
import tiktoken
from util.backend_apis import call_backend 

app = Flask(__name__)
app.config.from_object(app_config)
Session(app)
openai.api_key = os.getenv("OPENAI_API_KEY")

MAX_SECTION_LEN = 10
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
TEST_DATA_FILE="SolarAire-testing-data.csv"
EMBEDDING_FILE = "embedded_SolarAire_data_2.csv"
df = pd.read_csv(TEST_DATA_FILE, header=0)
df = df.set_index(["title", "heading"])

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    
    fname is the path to a CSV with exactly these named columns: 
        "title", "heading", "0", "1", ... up to the length of the embedding vectors.
    """
    
    df2 = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df2.columns if c != "title" and c != "heading"])
    return {
           (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df2.iterrows()
    }

document_embeddings = load_embeddings(EMBEDDING_FILE)

# This section is needed for url_for("foo", _external=True) to automatically
# generate http scheme when this sample is running on localhost,
# and to generate https scheme when it is deployed behind reversed proxy.
# See also https://flask.palletsprojects.com/en/1.0.x/deploying/wsgi-standalone/#proxy-setups
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

auth = identity.web.Auth(
    session=session,
    authority=app.config.get("AUTHORITY"),
    client_id=app.config["CLIENT_ID"],
    client_credential=app.config["CLIENT_SECRET"],
    )

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.03,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

@app.before_request
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = datetime.timedelta(minutes=20)
    session.modified = True

@app.route("/login")
def login():
    return render_template("login.html", version=identity.__version__, **auth.log_in(
        scopes=app_config.SCOPE,  # Have user consent scopes during log-in
        redirect_uri=url_for("auth_response", _external=True),  # Optional. If present, this absolute URL must match your app's redirect_uri registered in Azure Portal
        ))

@app.route(app_config.REDIRECT_PATH)
def auth_response():
    result = auth.complete_log_in(request.args)
    return render_template("auth_error.html", result=result) if "error" in result else redirect(url_for("index"))

@app.route("/logout")
def logout():
    return redirect(auth.log_out(url_for("index", _external=True)))

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    encoding = tiktoken.get_encoding(ENCODING)
    separator_len = len(encoding.encode(SEPARATOR))

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    #header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    header = "I'm a good assist.. \n\nContext:\n"""
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

def generate_prompt(question):
    if "SolarAire" or "solaraire" in question: 
        prompt = construct_prompt(
           question,
           document_embeddings,
           df
      )
    else:
        #prompt = "Answer the question as truthfully as possible, and if you're unsure of the answer, say ""Sorry, I don't know"". \n\n Q: " + question + "\n A:" 
        prompt = "I'm a good assist. \n\n Q: " + question + "\n A:" 

    return prompt

def callBackend(data):


@app.route("/", methods=("GET", "POST"))
def index():
    if not auth.get_user():
        return redirect(url_for("login"))

    if request.method == "POST":
        question = request.form["question"]
        response = openai.Completion.create(
            prompt=generate_prompt(question),
            **COMPLETIONS_API_PARAMS
        )
        return redirect(url_for("index", user=auth.get_user(), result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("index.html", user=auth.get_user(), result=result)
