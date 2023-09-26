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

scope = "playlist-read-private"
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
    "valence",
]


def parse_audio_features(audio_features_list: list[dict]) -> list[list[float]]:
    audio_features: list[list[float]] = []
    for af in audio_features_list:
        feature_vec = [af[feat] for feat in features]
        audio_features.append(feature_vec)
    return audio_features


def get_track_ids(tracks: dict) -> list[str]:
    return [t["track"]["id"] for t in tracks["items"] if t["track"]["id"]]


def get_playlist_track_data(playlist_id: str) -> tuple[list, list]:
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

with open("data/apreslist_ids.pkl", "wb") as f:
    pickle.dump(apreslist_ids, f)
with open("data/seriously_ids.pkl", "wb") as f:
    pickle.dump(seriously_ids, f)
with open("data/zzz_ids.pkl", "wb") as f:
    pickle.dump(zzz_ids, f)
with open("data/sheep_ids.pkl", "wb") as f:
    pickle.dump(sheep_ids, f)

np.save("data/apreslist_M.npy", apreslist_M)
np.save("data/seriously_M.npy", seriously_M)
np.save("data/zzz_M.npy", zzz_M)
np.save("data/sheep_M.npy", sheep_M)

with open("data/apreslist_ids.pkl", "rb") as f:
    apreslist_ids = pickle.load(f)
with open("data/seriously_ids.pkl", "rb") as f:
    seriously_ids = pickle.load(f)
with open("data/zzz_ids.pkl", "rb") as f:
    zzz_ids = pickle.load(f)
with open("data/sheep_ids.pkl", "rb") as f:
    sheep_ids = pickle.load(f)

apreslist_M = np.load("data/apreslist_M.npy", allow_pickle=False)
seriously_M = np.load("data/seriously_M.npy", allow_pickle=False)
zzz_M = np.load("data/zzz_M.npy", allow_pickle=False)
sheep_M = np.load("data/sheep_M.npy", allow_pickle=False)

energy_dim = features.index("energy")

n_bins = 20
fig, axs = plt.subplots(2, 2, tight_layout=True, sharex=True)
axs[0][0].hist(apreslist_M[:, energy_dim], bins=n_bins)
axs[0][0].set_title("Apreslist")
axs[0][1].hist(seriously_M[:, energy_dim], bins=n_bins)
axs[0][1].set_title("Seriously")
axs[1][0].hist(zzz_M[:, energy_dim], bins=n_bins)
axs[1][0].set_title("zzz")
axs[1][1].hist(sheep_M[:, energy_dim], bins=n_bins)
axs[1][1].set_title("Sheep")
fig.suptitle("Distribution of energy for different playlist")
fig.show()

from scipy.stats import norm


def fit_univariate_normal(data: NDArray):
    loc = np.mean(data)
    scale = np.std(data)
    return norm(loc, scale)


def plot_hist_and_dist(
    ax,
    data: NDArray,
    dist,
    nbins: int = 20,
    limit: tuple[float, float] | None = None,
    xlim: list[float] = [0, 1],
) -> None:
    ax.hist(data, bins=nbins, density=True)
    if limit:
        x = np.linspace(*limit, 100)
    else:
        x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 100)
    ax.plot(x, dist.pdf(x), "r-", lw=2, alpha=0.6, label="pdf")
    ax.set_xlim(xlim)


n_bins = 20
fig, axs = plt.subplots(2, 2, tight_layout=True)
dist_1 = fit_univariate_normal(apreslist_M[:, energy_dim])
plot_hist_and_dist(axs[0][0], apreslist_M[:, energy_dim], dist_1)
axs[0][0].set_title("Apreslist")
dist_2 = fit_univariate_normal(seriously_M[:, energy_dim])
plot_hist_and_dist(axs[0][1], seriously_M[:, energy_dim], dist_2)
axs[0][1].set_title("Seriously")
dist_3 = fit_univariate_normal(zzz_M[:, energy_dim])
plot_hist_and_dist(axs[1][0], zzz_M[:, energy_dim], dist_3)
axs[1][0].set_title("zzz")
dist_4 = fit_univariate_normal(sheep_M[:, energy_dim])
plot_hist_and_dist(axs[1][1], sheep_M[:, energy_dim], dist_4)
axs[1][1].set_title("Sheep")
fig.suptitle("Distribution of energy for different playlists")
plt.show()

norm_ll = (
    np.sum(dist_1.logpdf(apreslist_M[:, energy_dim]))
    + np.sum(dist_2.logpdf(seriously_M[:, energy_dim]))
    + np.sum(dist_3.logpdf(zzz_M[:, energy_dim]))
    + np.sum(dist_4.logpdf(sheep_M[:, energy_dim]))
)
print(f"combined log likelihood for normal distribution: {norm_ll:.2f}")

norm.nnlf(dist_1.args, apreslist_M[:, energy_dim])

from scipy.stats import beta
from scipy.optimize import minimize


def fit_beta(data: NDArray):
    params = beta.fit(data)
    return beta(*params)


def fit_beta_optimized(data: NDArray):
    def func(p: tuple, r: NDArray) -> float:
        return -np.sum(beta.logpdf(r, *p))

    params = minimize(
        func,
        x0=(1, 1),
        args=(data),
        bounds=((0, None), (0, None)),
    )

    return beta(*params.x)


def plot_hist_and_beta_dists(
    ax,
    data: NDArray,
    dist_fit,
    dist_optim,
    nbins: int = 20,
) -> None:
    ax.hist(data, bins=nbins, density=True)
    x = np.linspace(0, 1, 100)
    ax.plot(x, dist_fit.pdf(x), "r-", lw=2, alpha=0.6, label="fit pdf")
    ax.plot(x, dist_optim.pdf(x), "b-", lw=2, alpha=0.6, label="optimized pdf")
    ax.set_xlim([0, 1])
    ax.legend()


n_bins = 20
fig, axs = plt.subplots(2, 2, tight_layout=True)
dist_1_b = fit_beta(apreslist_M[:, energy_dim])
dist_1_o = fit_beta_optimized(apreslist_M[:, energy_dim])
plot_hist_and_beta_dists(axs[0][0], apreslist_M[:, energy_dim], dist_1_b, dist_1_o)
axs[0][0].set_title("Apreslist")
dist_2_b = fit_beta(seriously_M[:, energy_dim])
dist_2_o = fit_beta_optimized(seriously_M[:, energy_dim])
plot_hist_and_beta_dists(axs[0][1], seriously_M[:, energy_dim], dist_2_b, dist_2_o)
axs[0][1].set_title("Seriously")
dist_3_b = fit_beta(zzz_M[:, energy_dim])
dist_3_o = fit_beta_optimized(zzz_M[:, energy_dim])
plot_hist_and_beta_dists(axs[1][0], zzz_M[:, energy_dim], dist_3_b, dist_3_o)
axs[1][0].set_title("zzz")
dist_4_b = fit_beta(sheep_M[:, energy_dim])
dist_4_o = fit_beta_optimized(sheep_M[:, energy_dim])
plot_hist_and_beta_dists(axs[1][1], sheep_M[:, energy_dim], dist_4_b, dist_4_o)
axs[1][1].set_title("Sheep")
fig.suptitle("Distribution of energy for different playlists")
plt.show()

norm_ll = (
    np.sum(dist_1_b.logpdf(apreslist_M[:, energy_dim]))
    + np.sum(dist_2_b.logpdf(seriously_M[:, energy_dim]))
    + np.sum(dist_3_b.logpdf(zzz_M[:, energy_dim]))
    + np.sum(dist_4_b.logpdf(sheep_M[:, energy_dim]))
)
print(
    f"combined log likelihood for beta distribution with standard fitting: {norm_ll:.2f}"
)

norm_ll = (
    np.sum(dist_1_o.logpdf(apreslist_M[:, energy_dim]))
    + np.sum(dist_2_o.logpdf(seriously_M[:, energy_dim]))
    + np.sum(dist_3_o.logpdf(zzz_M[:, energy_dim]))
    + np.sum(dist_4_o.logpdf(sheep_M[:, energy_dim]))
)
print(
    f"combined log likelihood for beta distribution with optimized fitting: {norm_ll:.2f}"
)

print("Apreslist")
print(f"Normal: {np.sum(dist_1.logpdf(apreslist_M[:, energy_dim])):.2f}")
print(f"Beta standard fit: {np.sum(dist_1_b.logpdf(apreslist_M[:, energy_dim])):.2f}")
print(
    f"Beta optimized fit: {np.sum(dist_1_o.logpdf(apreslist_M[:, energy_dim])):.2f}\n"
)

print("Seriously")
print(f"Normal: {np.sum(dist_2.logpdf(seriously_M[:, energy_dim])):.2f}")
print(f"Beta standard fit: {np.sum(dist_2_b.logpdf(seriously_M[:, energy_dim])):.2f}")
print(
    f"Beta optimized fit: {np.sum(dist_2_o.logpdf(seriously_M[:, energy_dim])):.2f}\n"
)

print("zzz")
print(f"Normal: {np.sum(dist_3.logpdf(zzz_M[:, energy_dim])):.2f}")
print(f"Beta standard fit: {np.sum(dist_3_b.logpdf(zzz_M[:, energy_dim])):.2f}")
print(f"Beta optimized fit: {np.sum(dist_3_o.logpdf(zzz_M[:, energy_dim])):.2f}\n")

print("sheep")
print(f"Normal: {np.sum(dist_4.logpdf(sheep_M[:, energy_dim])):.2f}")
print(f"Beta standard fit: {np.sum(dist_4_b.logpdf(sheep_M[:, energy_dim])):.2f}")
print(f"Beta optimized fit: {np.sum(dist_4_o.logpdf(sheep_M[:, energy_dim])):.2f}\n")


n_bins = 20
fig, axs = plt.subplots(2, 2, tight_layout=True)
dist_1 = fit_beta(apreslist_M[:, dance_dim])
plot_hist_and_dist(axs[0][0], apreslist_M[:, dance_dim], dist_1, limit=(0, 1))
axs[0][0].set_title("Apreslist")
dist_2 = fit_beta(seriously_M[:, dance_dim])
plot_hist_and_dist(axs[0][1], seriously_M[:, dance_dim], dist_2, limit=(0, 1))
axs[0][1].set_title("Seriously")
dist_3 = fit_beta(zzz_M[:, dance_dim])
plot_hist_and_dist(axs[1][0], zzz_M[:, dance_dim], dist_3, limit=(0, 1))
axs[1][0].set_title("zzz")
dist_4 = fit_beta(sheep_M[:, dance_dim])
plot_hist_and_dist(axs[1][1], sheep_M[:, dance_dim], dist_4, limit=(0, 1))
axs[1][1].set_title("Sheep")
fig.suptitle("Distribution of danceability for different playlists")
plt.show()


def plot_feature(feat_name: str, nbins: int = 20) -> None:
    feat = features.index(feat_name)

    fig, axs = plt.subplots(2, 2, tight_layout=True, sharex=True)
    axs[0][0].hist(apreslist_M[:, feat], bins=nbins)
    axs[0][0].set_title("Apreslist")
    axs[0][1].hist(seriously_M[:, feat], bins=nbins)
    axs[0][1].set_title("Seriously")
    axs[1][0].hist(zzz_M[:, feat], bins=nbins)
    axs[1][0].set_title("zzz")
    axs[1][1].hist(sheep_M[:, feat], bins=nbins)
    axs[1][1].set_title("Sheep")
    fig.suptitle(f"Distribution of {feat_name} for different playlists")
    plt.show()


def fit_and_plot_beta(feat_name: str, nbins: int = 20) -> tuple:
    feat = features.index(feat_name)

    fig, axs = plt.subplots(2, 2, tight_layout=True)
    dist_1 = fit_beta(apreslist_M[:, feat])
    plot_hist_and_dist(
        axs[0][0], apreslist_M[:, feat], dist_1, limit=(0, 1), nbins=nbins
    )
    axs[0][0].set_title("Apreslist")
    dist_2 = fit_beta(seriously_M[:, feat])
    plot_hist_and_dist(
        axs[0][1], seriously_M[:, feat], dist_2, limit=(0, 1), nbins=nbins
    )
    axs[0][1].set_title("Seriously")
    dist_3 = fit_beta(zzz_M[:, feat])
    plot_hist_and_dist(axs[1][0], zzz_M[:, feat], dist_3, limit=(0, 1), nbins=nbins)
    axs[1][0].set_title("zzz")
    dist_4 = fit_beta(sheep_M[:, feat])
    plot_hist_and_dist(axs[1][1], sheep_M[:, feat], dist_4, limit=(0, 1), nbins=nbins)
    axs[1][1].set_title("Sheep")
    fig.suptitle(f"Distribution of {feat_name} for different playlists")
    plt.show()

    return dist_1, dist_2, dist_3, dist_4


from scipy.stats import skewnorm

def fit_skew_normal(data: NDArray):
    params = skewnorm.fit(data)
    return skewnorm(*params)


loudness_dim = features.index("loudness")
n_bins = 20
fig, axs = plt.subplots(2, 2, tight_layout=True)
dist_1 = fit_skew_normal(apreslist_M[:, loudness_dim])
plot_hist_and_dist(axs[0][0], apreslist_M[:, loudness_dim], dist_1, xlim=[-45,0])
axs[0][0].set_title("Apreslist")
dist_2 = fit_skew_normal(seriously_M[:, loudness_dim])
plot_hist_and_dist(axs[0][1], seriously_M[:, loudness_dim], dist_2, xlim=[-45,0])
axs[0][1].set_title("Seriously")
dist_3 = fit_skew_normal(zzz_M[:, loudness_dim])
plot_hist_and_dist(axs[1][0], zzz_M[:, loudness_dim], dist_3, xlim=[-45,0])
axs[1][0].set_title("zzz")
dist_4 = fit_skew_normal(sheep_M[:, loudness_dim])
plot_hist_and_dist(axs[1][1], sheep_M[:, loudness_dim], dist_4, xlim=[-45,0])
axs[1][1].set_title("Sheep")
fig.suptitle("Distribution of loudness for different playlists")
plt.show()

tempo_dim = features.index("tempo")
n_bins = 20
fig, axs = plt.subplots(2, 2, tight_layout=True)
dist_1 = fit_skew_normal(apreslist_M[:, tempo_dim])
plot_hist_and_dist(axs[0][0], apreslist_M[:, tempo_dim], dist_1, xlim=[0,200])
axs[0][0].set_title("Apreslist")
dist_2 = fit_skew_normal(seriously_M[:, tempo_dim])
plot_hist_and_dist(axs[0][1], seriously_M[:, tempo_dim], dist_2, xlim=[0,200])
axs[0][1].set_title("Seriously")
dist_3 = fit_skew_normal(zzz_M[:, tempo_dim])
plot_hist_and_dist(axs[1][0], zzz_M[:, tempo_dim], dist_3, xlim=[0,200])
axs[1][0].set_title("zzz")
dist_4 = fit_skew_normal(sheep_M[:, tempo_dim])
plot_hist_and_dist(axs[1][1], sheep_M[:, tempo_dim], dist_4, xlim=[0,200])
axs[1][1].set_title("Sheep")
fig.suptitle("Distribution of tempo for different playlists")
plt.show()

So to recap we have 6 potentially useful features in:
    * acousticness
    * danceability
    * energy
    * instrumentalness
    * loudness
    * valence

