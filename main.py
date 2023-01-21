import plexapi
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--baseurl', help='baseurl of the plex server', required=True)
parser.add_argument('--token', help='token to access the plex server', required=True)
args = parser.parse_args()

plex = plexapi.PlexServer(args.baseurl, args.token)
watch_history = plex.history()
shows = plex.library.section('shows').all()
movies = plex.library.section('movies').all()
shows_data = [show.summary for show in shows]
movies_data = [movie.summary for movie in movies]
data = shows_data + movies_data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data)
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
clusters = kmeans.predict(X)
prompt = input("Would you like suggestions for shows or movies? (shows/movies)")
try:
with open("suggestions.pickle", "rb") as f:
stored_data = pickle.load(f)
time_elapsed = datetime.now() - stored_data["timestamp"]
if time_elapsed.days < 15:
suggestions = stored_data["suggestions"]
else:
if prompt.lower() == "shows":
suggestions = [shows[i] for i in range(len(shows)) if clusters[i] == kmeans.predict(vectorizer.transform([watch_history[-1].summary]))[0]]
elif prompt.lower() == "movies":
suggestions = [movies[i] for i in range(len(movies)) if clusters[i] == kmeans.predict(vectorizer.transform([watch_history[-1].summary]))[0]]
else:
print("Invalid input. Please enter 'shows' or 'movies'.")
#update the stored suggestions
with open("suggestions.pickle", "wb") as f:
pickle.dump({"timestamp": datetime.now(), "suggestions": suggestions}, f)
except FileNotFoundError:
suggestions = [shows[i] for i in range(len(shows)) if clusters[i] == kmeans.predict(vectorizer.transform([watch_history[-1].summary]))[0]]
elif prompt.lower() == "movies":
suggestions = [movies[i] for i in range(len(movies)) if clusters[i] == kmeans.predict(vectorizer.transform([watch_history[-1].summary]))[0]]
else:
print("Invalid input. Please enter 'shows' or 'movies'.")
with open("suggestions.pickle", "wb") as f:
pickle.dump({"timestamp": datetime.now(), "suggestions": suggestions}, f)
print("Here are your suggestions:")
for suggestion in suggestions:
print(suggestion.title)
