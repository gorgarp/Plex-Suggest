import requests
import json
import pickle
from sklearn.cluster import KMeans
import numpy as np
import time
from flask import Flask, render_template, request

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
    print(watch_history_response.text)
    watch_history_data = json.loads(watch_history_response.text)
    watch_history_data = []
    for session in watch_history_data["MediaContainer"]["Metadata"]:
        watch_history_data.append([session["title"], session["year"], session["rating"]])
    return watch_history_data

def make_recommendations(username):
    # Fetch the watch history of the user
    user_watch_history = []
    for session in watch_history_data:
        if session["username"] == username:
            user_watch_history.append([session["title"], session["year"], session["rating"]])
    # Predict the cluster of the user's watch history
    cluster_id = kmeans.predict(user_watch_history)
    # Make recommendations based on the cluster centroid
    suggestions = []
    for title, year, rating in kmeans.cluster_centers_[cluster_id]:
        suggestions.append(f"{title} ({year})")
    return suggestions

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the watch history of all users
        watch_history_data = get_watch_history_of_all_users()
        try:
            # Load the clustering model
            kmeans = load_clustering_model()
        except FileNotFoundError:
            # Train the clustering model
            kmeans = train_clustering_model(watch_history_data)
        # Get the username
        username = request.form['username']
        # Make recommendations
        recommendations = make_recommend
if __name__ == '__main__':
    while True:
        watch_history_data = get_watch_history_of_all_users()
        try:
            kmeans = load_clustering_model()
        except FileNotFoundError:
            kmeans = train_clustering_model(watch_history_data)
        app.run(debug=True)
        time.sleep(3600) # Pause for 1 hour (3600 seconds) before updating the watch history data again
