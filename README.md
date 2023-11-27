# Simple backend Flask server + Index server for RAG

1. download the repo to a local folder
```
git clone https://github.com/LZ-Azure904/backend.git
```
2. start the index server on windows with port=5602 in the directory

```
$ python -m venv venv
$ venv\Scripts\activate
$ pip install -r requirements.txt
$ python .\iBAE2_index_server.py &
```
3. start the Flask web server on windows with port-5601 using a sepearated cmdline 
```
$ venv\Scripts\activate
$ pip install -r requirements.txt
$ python .\iBAE2_app.py &

```
