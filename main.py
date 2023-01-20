import plexapi
import pickle
from plexapi.server import PlexServer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timedelta

# Connect to the Plex server
baseurl = 'http://localhost:32400'
token = 'YOUR_TOKEN'
plex = PlexServer(baseurl, token)

# Load the dataset from disk (if it exists)
try:
    with open('dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    dataset = {}

# Prompt the user to select which they want suggestions on
user_input = input("Please enter 'movies' or 'shows' to get suggestions on: ")

if user_input.lower() == 'movies':
    # Get the list of movies from the server
    movies = plex.library.section('Movies')
    titles = [movie.title for movie in movies.all()]
    genres = [movie.genres for movie in movies.all()]
    studio = [movie.studio for movie in movies.all()]
    rating = [movie.rating for movie in movies.all()]
elif user_input.lower() == 'shows':
    # Get the list of TV shows from the server
    tvshows = plex.library.section('TV Shows')
    titles = [show.title for show in tvshows.all()]
    genres = [show.genres for show in tvshows.all()]
    tags = [show.tags for show in tvshows.all()]
    rating = [show.rating for show in tvshows.all()]
    popularity = [show.popularity for show in tvshows.all()]
else:
    print("Invalid input. Please enter 'movies' or 'shows'.")
    exit()

# Create a Tf-Idf vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer to the titles, genres, tags, rating and popularity
vectorizer.fit(titles + genres + studio + rating + popularity)

# Get the user's watch history
watch_history = plex.library.recentlyWatched()

# Get the titles of the shows in the watch history
watch_history_titles = [item.title for item in watch_history]
watch_history_genres = [item.genres for item in watch_history]
watch_history_studio = [item.studio for item in watch_history]
watch_history_rating = [item.rating for item in watch_history]
watch_history_popularity = [item.popularity for item in watch_history]

# Vectorize the watch history titles, genres, tags, rating and popularity
watch_history_vectors = vectorizer.transform(watch_history_titles + watch_history_genres + watch_history_studio + watch_history_rating + watch_history_popularity)

# Calculate the cosine similarity between the watch history titles, genres, tags, rating and popularity and all titles, genres, tags, rating and popularity
similarities = cosine_similarity(watch_history_vectors, vectorizer.transform(titles + genres + studio + rating + popularity))

# Get the indices of the most similar titles, genres, tags, rating and popularity
most_similar_indices = similarities.argsort()[:, -5:][:, ::-1]

# Print the most similar titles, genres, tags, rating and popularity
for i in most_similar_indices:
    print(titles[i])
    print(genres[i])
    print(studio[i])
    print(rating[i])
    print(popularity[i])

# Save the dataset to disk
with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)
