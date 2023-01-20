import plexapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

# Prompt the user for their Plex credentials
username = input("Enter your Plex username: ")
password = input("Enter your Plex password: ")
plex = plexapi.PlexServer(PLEX_URL, username=username, password=password)

try:
    # Load the data from a file
    with open("watch_history.pkl", "rb") as f:
        watch_history = pickle.load(f)
    with open("user_groups.pkl", "rb") as f:
        user_groups = pickle.load(f)
    with open("recommended_items.pkl", "rb") as f:
        recommended_items = pickle.load(f)
except:
    # Extract the show/movie metadata and user watch history from the data
    watch_history = plex.history()
    metadata = [h.metadata for h in watch_history]
    users = [h.user for h in watch_history]
    user_history = {
        user: [m.ratingKey for m in metadata if m.user == user] for user in set(users)
    }
    recommended_items = {}

    # Use KMeans clustering to group users based on their watch history
    model = KMeans(n_clusters=5)
    user_groups = model.fit_predict(user_history)

# Ask the user if they are trying to find a show or movie
media_type = input("Are you trying to find a show or movie? ")

# Analyze the metadata of shows/movies in each group to make recommendations
for group in set(user_groups):
    group_metadata = [
        m
        for i, m in enumerate(metadata)
        if user_groups[i] == group and m.type == media_type
    ]
    group_users = [user for i, user in enumerate(users) if user_groups[i] == group]

    # Extracting features for content-based filtering
    genres = [m.genres for m in group_metadata]
    actors = [m.actors for m in group_metadata]
    directors = [m.directors for m in group_metadata]
popularity = [m.popularity for m in group_metadata]
ratings = [m.rating for m in group_metadata]
releasedates = [m.releasedate for m in group_metadata]
tags = [m.tags for m in group_metadata]
year = [m.year for m in group_metadata]
features = np.array(
    [genres, actors, directors, popularity, ratings, releasedates, tags, year]
)

# Compute cosine similarity
cosine_sim = cosine_similarity(features)
content_based_recs = []
for i in range(len(group_metadata)):
    similar_indices = np.argsort(cosine_sim[i])[:-11:-1]
    similar_items = [group_metadata[i] for i in similar_indices]
    content_based_recs.append(similar_items)

    # Make recommendations to each user
    for i, user in enumerate(group_users):
        recs = content_based_recs[i]
        # Remove already watched items from recommendations
        recs = [rec for rec in recs if rec.ratingKey not in user_history[user]]
        # Remove items that have already been recommended
        recs = [rec for rec in recs if rec.ratingKey not in recommended_items.keys()]
        # Update the recommended_items dictionary
        for rec in recs:
            recommended_items[rec.ratingKey] = rec
        # Print the recommendations for the user
        print(f"Recommendations for {user}:")
        for rec in recs:
            print(rec.title)
    # Save the data to a file
    with open("watch_history.pkl", "wb") as f:
        pickle.dump(watch_history, f)
    with open("user_groups.pkl", "wb") as f:
        pickle.dump(user_groups, f)
    with open("recommended_items.pkl", "wb") as f:
        pickle.dump(recommended_items, f)
