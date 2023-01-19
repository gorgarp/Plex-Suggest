from flask import Flask, render_template, request
import requests
import json
import pickle
from sklearn.cluster import KMeans
import numpy as np
import time

app = Flask(__name__)

def train_clustering_model(watch_history_data):
    watch_history_data = np.array(watch_history_data)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(watch_history_data)
    with open('kmeans_model.pkl', 'wb') as file:
        pickle.dump(kmeans, file)
    return kmeans

def load_clustering_model():
    with open('kmeans_model.pkl', 'rb') as file:
        kmeans = pickle.load(file)
    return kmeans

def get_watch_history_of_all_users():
    watch_history_url = f"https://plex.tv/api/v2/users/accounts/sessions"
    headers = {
        "X-Plex-Token": "YOUR_PLEX_API_TOKEN"
    }
    watch_history_response = requests.get(watch_history_url, headers=headers)
    watch_history_data = json.loads(watch_history_response.text)
    watch_history = []
    for session in watch_history_data["MediaContainer"]["Metadata"]:
        watch_history.append([session["title"], session["year"], session["rating"]])
    return watch_history

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #get the watch history of all users
        watch_history_data = get_watch_history_of_all_users()

        # Load or train the clustering model
        try:
            kmeans = load_clustering_model()
        except FileNotFoundError:
            kmeans = train_clustering_model(watch_history_data)
        
        # Predict the cluster of the current user
        username = request.form['username']
        user_watch_history = []
        for session in watch_history_data:
            if session["username"] == username:
                user_watch_history.append([session["title"], session["year"], session["rating"]])
        user_watch_history = np.array(user_watch_history)
        cluster_id = kmeans.predict(user_watch_history)
        
        # Make suggestions
        suggestions = []
        for title, year, rating in kmeans.cluster_centers_[cluster_id]:
            suggestions.append(f"{title} ({year})")

        return render_template('index.html', suggestions=suggestions)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    while True:
        watch_history_data = get_watch_history
