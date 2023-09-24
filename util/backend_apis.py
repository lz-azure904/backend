import requests 
import json
import os
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session

client_id = os.getenv("BACKEND_CLIENT_ID")
client_secret = os.getenv("BACKEND_CLIENT_SECRET")
token_url = os.getenv("BACKEND_TOKEN_URL") # The token endpoint of the OIDC provider
backend_url = os.getenv("BACKEND_URL") 

def call_backend(user_request):
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)
    token = oauth.fetch_token(token_url=token_url, client_id=client_id, client_secret=client_secret)

    headers = {
    'Authorization': f'Bearer {token["access_token"]}',
    'Content-Type': 'application/json'
    }
    response = requests.get(backend_url, headers=headers)

    if response.status_code < 200 or response.status_code > 202:
        return "Could not find the answer"
    else:
        return response.choices[0].text




