# Simple backend Flask server + Index server for RAG

1. Install the requirements

   ```bash
   $ pip install -r requirements.txt
   ```
2. start the index server on windows with port=5602

```
$ python -m venv venv
$ venv\Scripts\activate
$ pip install -r requirements.txt
$ python .\iBAE2_index_server.py &
```
3. start the Flask web server on windows with port-5601
```
$ python -m venv venv
$ venv\Scripts\activate
$ pip install -r requirements.txt
$ python .\iBAE2_app.py &

```
