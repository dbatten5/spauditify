import numpy as np

import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from numpy.typing import NDArray

# the 4 downtempo playlist ids
APRESLIST_ID = "1Gbl0gYulJ5oXzCZWyR1Hr"
SERIOUSLY_ID = "12b7lLOydJ1rzgcp2H0uye"
ZZZ_ID = "0pWvAfI5sFdQ2DzaXXGLJ9"
SHEEP_ID = "5OJyVfvGLNO3o8rj0cHdOC"

scope = 'playlist-read-private'
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))

# grab all the tracks from the all playlists and return their audio features

features = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "key",
    "liveness",
    "loudness",
    "mode",
    "speechiness",
    "tempo",
    "valence"
]

def parse_audio_features(audio_features_list: list[dict]) -> list[list[float]]:
    audio_features: list[list[float]] = []
    for af in audio_features_list:
        feature_vec = [af[feat] for feat in features]
        audio_features.append(feature_vec)
    return audio_features

def get_track_ids(tracks: dict) -> list[str]:
    return [t["track"]["id"] for t in tracks["items"] if t["track"]["id"]]

def get_playlist_track_data(playlist_id: str) -> tuple[list,list]:
    track_ids: list[str] = []
    track_features: list[list[float]] = []
    track_results = sp.playlist(playlist_id, fields="tracks,next")
    tracks = track_results["tracks"]
    batch_track_ids = get_track_ids(tracks)
    track_ids.extend(batch_track_ids)
    audio_feature_results = sp.audio_features(batch_track_ids)
    batch_audio_features = parse_audio_features(audio_feature_results)
    track_features.extend(batch_audio_features)
    while tracks["next"]:
        tracks = sp.next(tracks)
        batch_track_ids = get_track_ids(tracks)
        track_ids.extend(batch_track_ids)
        audio_feature_results = sp.audio_features(batch_track_ids)
        batch_audio_features = parse_audio_features(audio_feature_results)
        track_features.extend(batch_audio_features)
    return track_ids, track_features


apreslist_ids, apreslist_feats = get_playlist_track_data(APRESLIST_ID)
seriously_ids, seriously_feats = get_playlist_track_data(SERIOUSLY_ID)
zzz_ids, zzz_feats = get_playlist_track_data(ZZZ_ID)
sheep_ids, sheep_feats = get_playlist_track_data(SHEEP_ID)

apreslist_M = np.array(apreslist_feats)
seriously_M = np.array(seriously_feats)
zzz_M = np.array(zzz_feats)
sheep_M = np.array(sheep_feats)

# to save calling the api again, save the ids and feats to the disk
import pickle
with open('data/apreslist_ids.pkl','wb') as f:
    pickle.dump(apreslist_ids, f)
with open('data/seriously_ids.pkl','wb') as f:
    pickle.dump(seriously_ids, f)
with open('data/zzz_ids.pkl','wb') as f:
    pickle.dump(zzz_ids, f)
with open('data/sheep_ids.pkl','wb') as f:
    pickle.dump(sheep_ids, f)

np.save("data/apreslist_M.npy", apreslist_M)
np.save("data/seriously_M.npy", seriously_M)
np.save("data/zzz_M.npy", zzz_M)
np.save("data/sheep_M.npy", sheep_M)

with open('data/apreslist_ids.pkl','rb') as f:
    apreslist_ids = pickle.load(f)
with open('data/seriously_ids.pkl','rb') as f:
    seriously_ids = pickle.load(f)
with open('data/zzz_ids.pkl','rb') as f:
    zzz_ids = pickle.load(f)
with open('data/sheep_ids.pkl','rb') as f:
    sheep_ids = pickle.load(f)

apreslist_M = np.load("data/apreslist_M.npy", allow_pickle=False)
seriously_M = np.load("data/seriously_M.npy", allow_pickle=False)
zzz_M = np.load("data/zzz_M.npy", allow_pickle=False)
sheep_M = np.load("data/sheep_M.npy", allow_pickle=False)

energy_dim = features.index("energy")

n_bins = 20
fig, axs = plt.subplots(2, 2, tight_layout=True, sharex=True)
axs[0][0].hist(apreslist_M[:,energy_dim], bins=n_bins)
axs[0][0].set_title("Apreslist")
axs[0][1].hist(seriously_M[:,energy_dim], bins=n_bins)
axs[0][1].set_title("Seriously")
axs[1][0].hist(zzz_M[:,energy_dim], bins=n_bins)
axs[1][0].set_title("zzz")
axs[1][1].hist(sheep_M[:,energy_dim], bins=n_bins)
axs[1][1].set_title("Sheep")
fig.suptitle("Distribution of energy for different playlist")
fig.show()

from scipy.stats import norm

def plot_uni_norm_histogram(ax, data: NDArray, nbins: int = 20) -> None:
    loc = np.mean(data)
    scale = np.std(data)
    ax.hist(data, bins=nbins, density=True)
    x = np.linspace(norm.ppf(0.01,loc,scale),
                    norm.ppf(0.99,loc,scale), 100)
    ax.plot(x, norm.pdf(x,loc,scale),
           'r-', lw=2, alpha=0.6, label='norm pdf')
    ax.set_xlim([0, 1])

n_bins = 20
fig, axs = plt.subplots(2, 2, tight_layout=True)
plot_uni_norm_histogram(axs[0][0], apreslist_M[:,energy_dim])
axs[0][0].set_title("Apreslist")
plot_uni_norm_histogram(axs[0][1], seriously_M[:,energy_dim])
axs[0][1].set_title("Seriously")
plot_uni_norm_histogram(axs[1][0], zzz_M[:,energy_dim])
axs[1][0].set_title("zzz")
plot_uni_norm_histogram(axs[1][1], sheep_M[:,energy_dim])
axs[1][1].set_title("Sheep")
fig.suptitle("Distribution of energy for different playlists")
fig.show()
