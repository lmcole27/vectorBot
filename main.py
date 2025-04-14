#import chromadb
#import pandas as pd
#from io import StringIO
from sentence_transformers import SentenceTransformer #type: ignore
#from cosine_similarity import compute_cosine_similarity
#from query import chatbot_query 

from openai import OpenAI
from flask import Flask, render_template, request, Response, stream_with_context, jsonify, session
from flask_cors import CORS # type: ignore
import json
import os
from dotenv import load_dotenv
import uuid
import atexit
from apscheduler.schedulers.background import BackgroundScheduler # type: ignore
from dataContext import getContext
from history import load_chat, add_message, save_chat, history_cleanup
import chromadb
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

load_dotenv()


# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
chromaClient = chromadb.PersistentClient(path="./chroma_data/")
try:
    collection = chromaClient.get_collection(name="amPlan_context")
    print(f"Collection 'amPlan_context' found with {collection.count()} documents")
except Exception as e:
    print(f"Error accessing collection: {e}")
    print("Creating new collection...")
    collection = chromaClient.create_collection(
        name="amPlan_context",
        metadata={"hnsw:space": "cosine"}
    )

#API INFO
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], organization=os.environ['ORGANIZATION'], project=os.environ['PROJECT'])
#chat_history = load_chat()

#CREATE WEBAPP
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.environ['FLASK_SECRET_KEY']

def generate_response(question: str, chat_history, context):
    # Send API request to ChatGPT and recieve resonse
    print(context)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}, 
                  {"role": "system", "content": f"consider the conversation context {chat_history}"},
                  {"role": "system", "content": context},
                  {"role": "system", "content":"provide response with HTML tags but no header."
                   }],
        stream=True,
    )

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            ans = chunk.choices[0].delta.content
            yield ans

    
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/api/set_session', methods=['GET','POST'])
def set_session():
    token = uuid.uuid4()  # unique guest ID
    token_str = str(token)
    session['token'] = token_str  # Store the token in session
    return jsonify(success=True, token=token_str)

@app.route('/api/get_session', methods=['GET', 'POST'])
def get_session():
    id = session.get('token', None)
    if id is None:
        return jsonify({"error": "No session found"}), 404
    return jsonify(success=True, token=id)

@app.route('/generate', methods=['POST'])
def generate():
    question = request.form.get('question')
    if not question:
        return "Please provide a question", 400
    # Get session token
    session_token = session.get('token')
    if not session_token:
        return "Session token not found", 400    

    # Load chat history for the user and save question to chat history
    chat_history = load_chat(session_token)
    add_message(chat_history, "user", question)
    save_chat(chat_history, session_token)
    
    # Get Context from source data
    context = getContext(model=model, collection=collection, query=question)

    def generate():
        for chunk in generate_response(question, chat_history, context):
            yield chunk

    return Response(stream_with_context(generate()), mimetype='text/plain')


# Example Usage:
# user_input = "Which industries have the highest insurance rates?"
# response = chatbot_query(user_input)
# print(response)

@app.route('/api/endpoint', methods=['POST'])
def receive_post():
    try:
        # Get JSON data from the request
        data = request.get_json()
        response = data["message"]

        # Get session token
        session_token = session.get('token')
        if not session_token:
            return jsonify({"error": "Session token not found"}), 400

        # Load chat history for the user
        chat_history = load_chat(session_token)

        # Save response to chat_history
        add_message(chat_history, "assistant", response)
        save_chat(chat_history, session_token)

        # Respond back to client
        return jsonify({"message": "Data received successfully", "received": data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Schedule job
scheduler = BackgroundScheduler()
scheduler.add_job(func=history_cleanup, trigger="interval", hours=1)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

#RUN THE WEBAPP
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


    