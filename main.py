import plexapi
from datetime
import datetime
from sklearn.feature_extraction.text
import TfidfVectorizer
from sklearn.cluster
import KMeans
import pickle
import argparse

# parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--baseurl', help = 'baseurl of the plex server', required = True)
parser.add_argument('--token', help = 'token to access the plex server', required = True)
args = parser.parse_args()

# Use token - based authentication to log into a Plex server
plex = plexapi.PlexServer(args.baseurl, args.token)

# Pull the user 's watch history
watch_history = plex.history()

# Fetch all the shows and movies from the server
shows = plex.library.section('shows').all()
movies = plex.library.section('movies').all()

# Categorize shows and movies based on their metadata
shows_data = [show.summary
    for show in shows
]
movies_data = [movie.summary
    for movie in movies
]

# Combine the data of shows and movies
data = shows_data + movies_data

# Extract features from the data using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = 'english')
X = vectorizer.fit_transform(data)

# Use k - means clustering to group the data into clusters
kmeans = KMeans(n_clusters = 5)
kmeans.fit(X)

# Assign each show / movie to a cluster
clusters = kmeans.predict(X)

# Prompt the user
for suggestions
prompt = input("Would you like suggestions for shows or movies? (shows/movies)")

# Check
if the suggestions file exists
try:
with open("suggestions.pickle", "rb") as f:
    stored_data = pickle.load(f)# Check the time elapsed since the last suggestion
time_elapsed = datetime.now() - stored_data["timestamp"]
if time_elapsed.days < 15: #Suggest the same shows / movies
suggestions = stored_data["suggestions"]
else :#Update the suggestions
if prompt.lower() == "shows":
    suggestions = [shows[i]
        for i in range(len(shows)) if clusters[i] == kmeans.predict(vectorizer.transform([watch_history[-1].summary]))[0]
    ]
elif prompt.lower() == "movies":
    suggestions = [movies[i]
        for i in range(len(movies)) if clusters[i] == kmeans.predict(vectorizer.transform([watch_history[-1].summary]))[0]
    ]
else :
    print("Invalid input. Please enter 'shows' or 'movies'.")# update the stored suggestions
with open("suggestions.pickle", "wb") as f:
    pickle.dump({
        "timestamp": datetime.now(),
        "suggestions": suggestions
    }, f)
except FileNotFoundError:
    suggestions = [shows[i]
        for i in range(len(shows)) if clusters[i] == kmeans.predict(vectorizer.transform([watch_history[-1].summary]))[0]
    ]
elif prompt.lower() == "movies":
    suggestions = [movies[i]
        for i in range(len(movies)) if clusters[i] == kmeans.predict(vectorizer.transform([watch_history[-1].summary]))[0]
    ]
else :
    print("Invalid input. Please enter 'shows' or 'movies'.")
with open("suggestions.pickle", "wb") as f:
    pickle.dump({
        "timestamp": datetime.now(),
        "suggestions": suggestions
    }, f)

# Print the suggestions
print("Here are your suggestions:")
for suggestion in suggestions:
    print(suggestion.title)
