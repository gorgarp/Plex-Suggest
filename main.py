from flask import Flask, render_template, request
import requests
import json
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user's watch history
        username = request.form['username']
        watch_history_url = f"https://plex.tv/api/v2/users/accounts/sessions"
        headers = {
            "X-Plex-Token": "YOUR_PLEX_API_TOKEN"
        }
        params = {
            'username': username
        }
        watch_history_response = requests.get(watch_history_url, headers=headers, params=params)
        watch_history_data = json.loads(watch_history_response.text)

        # Cluster the users based on their watch history
        watch_history_data = []
        for session in watch_history_data["MediaContainer"]["Metadata"]:
            watch_history_data.append([session["title"], session["year"], session["rating"]])
        watch_history_data = np.array(watch_history_data)
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(watch_history_data)

        # Make suggestions based on the user's cluster
        cluster_id = kmeans.predict(watch_history_data)
        suggestions = []
        for title, year, rating in kmeans.cluster_centers_[cluster_id]:
            suggestions.append(f"{title} ({year})")

        return render_template('index.html', suggestions=suggestions)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
