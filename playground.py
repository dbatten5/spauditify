import numpy as np
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyOAuth


APRESLIST_ID = "1Gbl0gYulJ5oXzCZWyR1Hr"
SERIOUSLY_ID = "12b7lLOydJ1rzgcp2H0uye"
ZZZ_ID = "0pWvAfI5sFdQ2DzaXXGLJ9"
SHEEP_ID = "5OJyVfvGLNO3o8rj0cHdOC"


scope = "playlist-read-private"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))


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


def get_track_data(tracks: dict) -> list[dict]:
    data: list[dict] = []
    for t in tracks["items"]:
        tr = t["track"]
        if not tr["id"]:
            continue
        datum = {
            "id": tr["id"],
            "name": tr["name"],
            "artist": tr["artists"][0]["name"],
            "album": tr["album"]["name"],
            "uri": tr["uri"],
        }
        data.append(datum)
    return data


def get_playlist_track_data(playlist_id: str) -> tuple[list, list]:
    track_data: list[dict] = []
    track_features: list[list[float]] = []
    track_results = sp.playlist(playlist_id, fields="tracks,next")
    tracks = track_results["tracks"]
    batch_track_data = get_track_data(tracks)
    track_data.extend(batch_track_data)
    audio_feature_results = sp.audio_features([td["id"] for td in batch_track_data])
    batch_audio_features = parse_audio_features(audio_feature_results)
    track_features.extend(batch_audio_features)
    while tracks["next"]:
        tracks = sp.next(tracks)
        batch_track_data = get_track_data(tracks)
        track_data.extend(batch_track_data)
        audio_feature_results = sp.audio_features([td["id"] for td in batch_track_data])
        batch_audio_features = parse_audio_features(audio_feature_results)
        track_features.extend(batch_audio_features)
    return track_data, track_features


apreslist_track_data, apreslist_feats = get_playlist_track_data(APRESLIST_ID)
seriously_track_data, seriously_feats = get_playlist_track_data(SERIOUSLY_ID)
zzz_track_data, zzz_feats = get_playlist_track_data(ZZZ_ID)
sheep_track_data, sheep_feats = get_playlist_track_data(SHEEP_ID)

apreslist_M = np.array(apreslist_feats)
seriously_M = np.array(seriously_feats)
zzz_M = np.array(zzz_feats)
sheep_M = np.array(sheep_feats)


import pickle

with open("data/apreslist_track_data.pkl", "wb") as f:
    pickle.dump(apreslist_track_data, f)
with open("data/seriously_track_data.pkl", "wb") as f:
    pickle.dump(seriously_track_data, f)
with open("data/zzz_track_data.pkl", "wb") as f:
    pickle.dump(zzz_track_data, f)
with open("data/sheep_track_data.pkl", "wb") as f:
    pickle.dump(sheep_track_data, f)

np.save("data/apreslist_M.npy", apreslist_M)
np.save("data/seriously_M.npy", seriously_M)
np.save("data/zzz_M.npy", zzz_M)
np.save("data/sheep_M.npy", sheep_M)


import pickle

with open("data/apreslist_track_data.pkl", "rb") as f:
    apreslist_track_data = pickle.load(f)
with open("data/seriously_track_data.pkl", "rb") as f:
    seriously_track_data = pickle.load(f)
with open("data/zzz_track_data.pkl", "rb") as f:
    zzz_track_data = pickle.load(f)
with open("data/sheep_track_data.pkl", "rb") as f:
    sheep_track_data = pickle.load(f)

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
fig.suptitle("Distribution of energy for different playlists")
plt.show()


from scipy.stats import norm
from numpy.typing import NDArray


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
print(f"total log likelihood for normal distribution: {norm_ll:.2f}")


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
    x = np.linspace(0.001, 0.999, 100)
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

print("Sheep")
print(f"Normal: {np.sum(dist_4.logpdf(sheep_M[:, energy_dim])):.2f}")
print(f"Beta standard fit: {np.sum(dist_4_b.logpdf(sheep_M[:, energy_dim])):.2f}")
print(f"Beta optimized fit: {np.sum(dist_4_o.logpdf(sheep_M[:, energy_dim])):.2f}\n")


models = [[dist_1_b, dist_2_b, dist_3_b, dist_4_b]]


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


plot_feature("danceability")


def fit_and_plot_beta(feat_name: str, nbins: int = 20) -> tuple:
    feat = features.index(feat_name)

    fig, axs = plt.subplots(2, 2, tight_layout=True)
    dist_1 = fit_beta(apreslist_M[:, feat])
    plot_hist_and_dist(
        axs[0][0], apreslist_M[:, feat], dist_1, limit=(0.01, 0.99), nbins=nbins
    )
    axs[0][0].set_title("Apreslist")
    dist_2 = fit_beta(seriously_M[:, feat])
    plot_hist_and_dist(
        axs[0][1], seriously_M[:, feat], dist_2, limit=(0.01, 0.99), nbins=nbins
    )
    axs[0][1].set_title("Seriously")
    dist_3 = fit_beta(zzz_M[:, feat])
    plot_hist_and_dist(
        axs[1][0], zzz_M[:, feat], dist_3, limit=(0.01, 0.99), nbins=nbins
    )
    axs[1][0].set_title("zzz")
    dist_4 = fit_beta(sheep_M[:, feat])
    plot_hist_and_dist(
        axs[1][1], sheep_M[:, feat], dist_4, limit=(0.01, 0.99), nbins=nbins
    )
    axs[1][1].set_title("Sheep")
    fig.suptitle(f"Distribution of {feat_name} for different playlists")
    plt.show()

    return dist_1, dist_2, dist_3, dist_4


db_1, db_2, db_3, db_4 = fit_and_plot_beta("danceability")


models.append([db_1, db_2, db_3, db_4])


plot_feature("acousticness")


ab_1, ab_2, ab_3, ab_4 = fit_and_plot_beta("acousticness")


models.append([ab_1, ab_2, ab_3, ab_4])


print(f"Seriously alpha value: {ab_2.args[0]:.2f}, beta value: {ab_2.args[1]:.2f}")
print(f"zzz alpha value: {ab_3.args[0]:.2f}, beta value: {ab_3.args[1]:.2f}")
print(f"Sheep alpha value: {ab_4.args[0]:.2f}, beta value: {ab_4.args[1]:.2f}")


plot_feature("instrumentalness")


ib_1, ib_2, ib_3, ib_4 = fit_and_plot_beta("instrumentalness")


models.append([ib_1, ib_2, ib_3, ib_4])


plot_feature("loudness")


from scipy.stats import skewnorm


def fit_skew_normal(data: NDArray):
    params = skewnorm.fit(data)
    return skewnorm(*params)


loudness_dim = features.index("loudness")
n_bins = 20
fig, axs = plt.subplots(2, 2, tight_layout=True)
dist_1 = fit_skew_normal(apreslist_M[:, loudness_dim])
plot_hist_and_dist(axs[0][0], apreslist_M[:, loudness_dim], dist_1, xlim=[-45, 0])
axs[0][0].set_title("Apreslist")
dist_2 = fit_skew_normal(seriously_M[:, loudness_dim])
plot_hist_and_dist(axs[0][1], seriously_M[:, loudness_dim], dist_2, xlim=[-45, 0])
axs[0][1].set_title("Seriously")
dist_3 = fit_skew_normal(zzz_M[:, loudness_dim])
plot_hist_and_dist(axs[1][0], zzz_M[:, loudness_dim], dist_3, xlim=[-45, 0])
axs[1][0].set_title("zzz")
dist_4 = fit_skew_normal(sheep_M[:, loudness_dim])
plot_hist_and_dist(axs[1][1], sheep_M[:, loudness_dim], dist_4, xlim=[-45, 0])
axs[1][1].set_title("Sheep")
fig.suptitle("Distribution of loudness for different playlists")
plt.show()


models.append([dist_1, dist_2, dist_3, dist_4])


plot_feature("valence")


vb_1, vb_2, vb_3, vb_4 = fit_and_plot_beta("valence")


models.append([vb_1, vb_2, vb_3, vb_4])


with open("models/model_matrix.pkl", "wb") as f:
    pickle.dump(models, f)


plot_feature("liveness")


plot_feature("speechiness")


plot_feature("key")


plot_feature("tempo")


tempo_dim = features.index("tempo")
n_bins = 20
fig, axs = plt.subplots(2, 2, tight_layout=True)
dist_1 = fit_skew_normal(apreslist_M[:, tempo_dim])
plot_hist_and_dist(axs[0][0], apreslist_M[:, tempo_dim], dist_1, xlim=[0, 200])
axs[0][0].set_title("Apreslist")
dist_2 = fit_skew_normal(seriously_M[:, tempo_dim])
plot_hist_and_dist(axs[0][1], seriously_M[:, tempo_dim], dist_2, xlim=[0, 200])
axs[0][1].set_title("Seriously")
dist_3 = fit_skew_normal(zzz_M[:, tempo_dim])
plot_hist_and_dist(axs[1][0], zzz_M[:, tempo_dim], dist_3, xlim=[0, 200])
axs[1][0].set_title("zzz")
dist_4 = fit_skew_normal(sheep_M[:, tempo_dim])
plot_hist_and_dist(axs[1][1], sheep_M[:, tempo_dim], dist_4, xlim=[0, 200])
axs[1][1].set_title("Sheep")
fig.suptitle("Distribution of tempo for different playlists")
plt.show()


plot_feature("mode")


with open("models/model_matrix.pkl", "rb") as f:
    models = pickle.load(f)


useful_dims = (2, 1, 0, 3, 6, 10)
all_tracks_M = np.vstack((apreslist_M, seriously_M, zzz_M, sheep_M))[:, useful_dims]


np.save("data/all_tracks_M", all_tracks_M)


all_tracks_M = np.load("data/all_tracks_M")


apreslist_ll = np.hstack(
    [models[ii][0].logpdf(all_tracks_M[:, ii]).reshape(-1, 1) for ii in range(6)]
)
seriously_ll = np.hstack(
    [models[ii][1].logpdf(all_tracks_M[:, ii]).reshape(-1, 1) for ii in range(6)]
)
zzz_ll = np.hstack(
    [models[ii][2].logpdf(all_tracks_M[:, ii]).reshape(-1, 1) for ii in range(6)]
)
sheep_ll = np.hstack(
    [models[ii][3].logpdf(all_tracks_M[:, ii]).reshape(-1, 1) for ii in range(6)]
)

all_ll = np.hstack(
    (
        np.sum(apreslist_ll, 1).reshape(-1, 1),
        np.sum(seriously_ll, 1).reshape(-1, 1),
        np.sum(zzz_ll, 1).reshape(-1, 1),
        np.sum(sheep_ll, 1).reshape(-1, 1),
    )
)

np.argmax(all_ll, 1)[:10]

wt = apreslist_track_data[5]
print(f"{wt['name']} by {wt['artist']}")

t = all_tracks_M[5]
bx = np.linspace(0.001, 0.999, 100)
gx = np.linspace(-45, 0, 100)
fig, axs = plt.subplots(2, 3, tight_layout=True)
axs[0][0].axvline(t[0], color="g", linestyle="--")
axs[0][0].plot(bx, models[0][0].pdf(bx), "r-", lw=2, alpha=0.6, label="Apreslist")
axs[0][0].plot(bx, models[0][1].pdf(bx), "b-", lw=2, alpha=0.6, label="Seriously")
axs[0][0].set_title("Energy")
axs[0][1].axvline(t[1], color="g", linestyle="--")
axs[0][1].plot(bx, models[1][0].pdf(bx), "r-", lw=2, alpha=0.6, label="Apreslist")
axs[0][1].plot(bx, models[1][1].pdf(bx), "b-", lw=2, alpha=0.6, label="Seriously")
axs[0][1].set_title("Danceability")
axs[0][2].axvline(t[2], color="g", linestyle="--")
axs[0][2].plot(bx, models[2][0].pdf(bx), "r-", lw=2, alpha=0.6, label="Apreslist")
axs[0][2].plot(bx, models[2][1].pdf(bx), "b-", lw=2, alpha=0.6, label="Seriously")
axs[0][2].set_title("Acousticness")
axs[0][2].legend()
axs[1][0].axvline(t[3], color="g", linestyle="--")
axs[1][0].plot(bx, models[3][0].pdf(bx), "r-", lw=2, alpha=0.6, label="Apreslist")
axs[1][0].plot(bx, models[3][1].pdf(bx), "b-", lw=2, alpha=0.6, label="Seriously")
axs[1][0].set_title("Instrumentalness")
axs[1][1].axvline(t[4], color="g", linestyle="--")
axs[1][1].plot(gx, models[4][0].pdf(gx), "r-", lw=2, alpha=0.6, label="Apreslist")
axs[1][1].plot(gx, models[4][1].pdf(gx), "b-", lw=2, alpha=0.6, label="Seriously")
axs[1][1].set_title("Loudness")
axs[1][2].axvline(t[5], color="g", linestyle="--")
axs[1][2].plot(bx, models[5][0].pdf(bx), "r-", lw=2, alpha=0.6, label="Apreslist")
axs[1][2].plot(bx, models[5][1].pdf(bx), "b-", lw=2, alpha=0.6, label="Seriously")
axs[1][2].set_title("Valence")
fig.suptitle("Distribution of energy for different playlists")
plt.show()

y_pred = np.argmax(all_ll, 1)
y_true = np.hstack(
    (
        np.zeros(len(apreslist_track_data)),
        np.ones(len(seriously_track_data)),
        np.ones(len(zzz_track_data)) * 2,
        np.ones(len(sheep_track_data)) * 3,
    )
)

from sklearn import metrics

cm = metrics.confusion_matrix(y_true, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Apreslist", "Seriously", "zzz", "Sheep"]
)

cm_display.plot()
plt.show()

print(f"Total tracks: {len(y_pred)}")
print(f"Correctly placed tracks: {np.sum(y_pred==y_true)}")
print(f"Incorrectly placed tracks: {len(y_pred)-np.sum(y_pred==y_true)}")

cur_apreslist = all_ll[: len(apreslist_track_data), :]
apres_to_move_ii = np.where(np.argmax(cur_apreslist, 1) == 2)[0]
for ii in apres_to_move_ii:
    t = apreslist_track_data[ii]
    print(f"{t['name']} by {t['artist']}")


def get_playlist_index(pl_name: str) -> int:
    if pl_name.lower() == "apreslist":
        return 0
    if pl_name.lower() == "seriously":
        return 1
    if pl_name.lower() == "zzz":
        return 2
    if pl_name.lower() == "sheep":
        return 3
    return 0


def plot_track_against_playlists(track_index: int, pl_1: str, pl_2: str) -> None:
    pl_1_i = get_playlist_index(pl_1)
    pl_2_i = get_playlist_index(pl_2)
    t = all_tracks_M[track_index]
    bx = np.linspace(0.001, 0.999, 100)
    gx = np.linspace(-45, 0, 100)
    fig, axs = plt.subplots(2, 3, tight_layout=True)
    axs[0][0].axvline(t[0], color="g", linestyle="--")
    axs[0][0].plot(bx, models[0][pl_1_i].pdf(bx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[0][0].plot(bx, models[0][pl_2_i].pdf(bx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[0][0].set_title("Energy")
    axs[0][1].axvline(t[1], color="g", linestyle="--")
    axs[0][1].plot(bx, models[1][pl_1_i].pdf(bx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[0][1].plot(bx, models[1][pl_2_i].pdf(bx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[0][1].set_title("Danceability")
    axs[0][2].axvline(t[2], color="g", linestyle="--")
    axs[0][2].plot(bx, models[2][pl_1_i].pdf(bx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[0][2].plot(bx, models[2][pl_2_i].pdf(bx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[0][2].set_title("Acousticness")
    axs[0][2].legend()
    axs[1][0].axvline(t[3], color="g", linestyle="--")
    axs[1][0].plot(bx, models[3][pl_1_i].pdf(bx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[1][0].plot(bx, models[3][pl_2_i].pdf(bx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[1][0].set_title("Instrumentalness")
    axs[1][1].axvline(t[4], color="g", linestyle="--")
    axs[1][1].plot(gx, models[4][pl_1_i].pdf(gx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[1][1].plot(gx, models[4][pl_2_i].pdf(gx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[1][1].set_title("Loudness")
    axs[1][2].axvline(t[5], color="g", linestyle="--")
    axs[1][2].plot(bx, models[5][pl_1_i].pdf(bx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[1][2].plot(bx, models[5][pl_2_i].pdf(bx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[1][2].set_title("Valence")
    fig.suptitle("Distribution of energy for different playlists")
    plt.show()


def get_all_ll(energy_weight: float = 1.0) -> NDArray:
    all_tracks_M_weighted = all_tracks_M.copy()
    all_tracks_M_weighted[:, 0] *= energy_weight
    apreslist_ll = np.hstack(
        [
            models[ii][0].logpdf(all_tracks_M_weighted[:, ii]).reshape(-1, 1)
            for ii in range(6)
        ]
    )
    seriously_ll = np.hstack(
        [
            models[ii][1].logpdf(all_tracks_M_weighted[:, ii]).reshape(-1, 1)
            for ii in range(6)
        ]
    )
    zzz_ll = np.hstack(
        [
            models[ii][2].logpdf(all_tracks_M_weighted[:, ii]).reshape(-1, 1)
            for ii in range(6)
        ]
    )
    sheep_ll = np.hstack(
        [
            models[ii][3].logpdf(all_tracks_M_weighted[:, ii]).reshape(-1, 1)
            for ii in range(6)
        ]
    )
    all_ll = np.hstack(
        (
            np.sum(apreslist_ll, 1).reshape(-1, 1),
            np.sum(seriously_ll, 1).reshape(-1, 1),
            np.sum(zzz_ll, 1).reshape(-1, 1),
            np.sum(sheep_ll, 1).reshape(-1, 1),
        )
    )
    return all_ll


import pandas as pd

all_df = pd.DataFrame(
    apreslist_track_data + seriously_track_data + zzz_track_data + sheep_track_data
)
X = np.vstack((apreslist_M, seriously_M, zzz_M, sheep_M))
X_df = pd.DataFrame(X, columns=features)

all_df = pd.concat((all_df, X_df), axis=1)

playlist_col = (
    ["Apreslist"] * len(apreslist_track_data)
    + ["Seriously"] * len(seriously_track_data)
    + ["zzz"] * len(zzz_track_data)
    + ["Sheep"] * len(sheep_track_data)
)
all_df["playlist"] = playlist_col

all_df.to_csv("data/all_df.csv")

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X)

labels = kmeans.labels_

plt.scatter(np.arange(len(labels)), labels)
plt.show()

plt.scatter(np.arange(len(labels)), labels)
plt.xlabel("Track index")
plt.ylabel("Playlist assignment")
plt.title("Assignment of tracks from KMeans")
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, all_df["playlist"], test_size=0.2, random_state=42
)

y = all_df["playlist"]

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)

clf.fit(X, y)

clf.score(X, y)

i = np.where(clf.predict(X) != y)[0]

candidates = all_df.iloc[i][["name", "artist", "playlist"]]
candidates["predicted playlist"] = clf.predict(X)[i]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf_1 = DecisionTreeClassifier(random_state=1)
clf_1.fit(X_train, y_train)
clf_1.score(X, y)


def get_candidates(
    tree: DecisionTreeClassifier, X: NDArray, y: NDArray | pd.Series
) -> pd.DataFrame:
    i = np.where(tree.predict(X) != y)[0]
    candidates = all_df.iloc[i][["name", "artist", "playlist"]]
    candidates["suggested playlist"] = tree.predict(X)[i]
    return candidates


clf_2 = DecisionTreeClassifier(random_state=2, max_depth=10)
clf_2.fit(X, y)
clf_2.score(X, y)
candidates = get_candidates(clf_2, X, y)


def plot_track_against_playlists_new(ts: pd.Series) -> None:
    pl_1 = ts["playlist"]
    pl_2 = ts["suggested playlist"]
    pl_1_i = get_playlist_index(pl_1)
    pl_2_i = get_playlist_index(pl_2)
    t = all_tracks_M[ts.name]
    bx = np.linspace(0.001, 0.999, 100)
    gx = np.linspace(-45, 0, 100)
    fig, axs = plt.subplots(2, 3, tight_layout=True, figsize=(14, 7))
    axs[0][0].axvline(t[0], color="g", linestyle="--")
    axs[0][0].plot(bx, models[0][pl_1_i].pdf(bx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[0][0].plot(bx, models[0][pl_2_i].pdf(bx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[0][0].set_title("Energy")
    axs[0][1].axvline(t[1], color="g", linestyle="--")
    axs[0][1].plot(bx, models[1][pl_1_i].pdf(bx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[0][1].plot(bx, models[1][pl_2_i].pdf(bx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[0][1].set_title("Danceability")
    axs[0][2].axvline(t[2], color="g", linestyle="--")
    axs[0][2].plot(bx, models[2][pl_1_i].pdf(bx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[0][2].plot(bx, models[2][pl_2_i].pdf(bx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[0][2].set_title("Acousticness")
    axs[0][2].legend()
    axs[1][0].axvline(t[3], color="g", linestyle="--")
    axs[1][0].plot(bx, models[3][pl_1_i].pdf(bx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[1][0].plot(bx, models[3][pl_2_i].pdf(bx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[1][0].set_title("Instrumentalness")
    axs[1][1].axvline(t[4], color="g", linestyle="--")
    axs[1][1].plot(gx, models[4][pl_1_i].pdf(gx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[1][1].plot(gx, models[4][pl_2_i].pdf(gx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[1][1].set_title("Loudness")
    axs[1][2].axvline(t[5], color="g", linestyle="--")
    axs[1][2].plot(bx, models[5][pl_1_i].pdf(bx), "r-", lw=2, alpha=0.6, label=pl_1)
    axs[1][2].plot(bx, models[5][pl_2_i].pdf(bx), "b-", lw=2, alpha=0.6, label=pl_2)
    axs[1][2].set_title("Valence")
    fig.suptitle(
        f"Feature values for {ts['name']} compared to distributions for {pl_1} and {pl_2}"
    )
    plt.show()


plot_track_against_playlists_new(candidates.loc[12])

indices = np.random.rand(len(all_df)) < 0.2



print("Apreslist")
print(f"Normal: {np.sum(en[0].logpdf(apr_df['energy'])):.2f}")
print(f"Beta standard fit: {np.sum(eb_f[0].logpdf(apr_df['energy'])):.2f}")
print(f"Beta optimized fit: {np.sum(eb_o[0].logpdf(apr_df['energy'])):.2f}\n")

print("Seriously")
print(f"Normal: {np.sum(en[1].logpdf(ser_df['energy'])):.2f}")
print(f"Beta standard fit: {np.sum(eb_f[1].logpdf(ser_df['energy'])):.2f}")
print(f"Beta optimized fit: {np.sum(eb_o[1].logpdf(ser_df['energy'])):.2f}\n")

print("zzz")
print(f"Normal: {np.sum(en[2].logpdf(zzz_df['energy'])):.2f}")
print(f"Beta standard fit: {np.sum(eb_f[2].logpdf(zzz_df['energy'])):.2f}")
print(f"Beta optimized fit: {np.sum(eb_o[2].logpdf(zzz_df['energy'])):.2f}\n")

print("Sheep")
print(f"Normal: {np.sum(en[3].logpdf(she_df['energy'])):.2f}")
print(f"Beta standard fit: {np.sum(eb_f[3].logpdf(she_df['energy'])):.2f}")
print(f"Beta optimized fit: {np.sum(eb_o[3].logpdf(she_df['energy'])):.2f}\n")

db = []

def fit_and_plot_beta(feature: str) -> list:
    dists = []
    g = sns.FacetGrid(all_df, col="playlist", col_wrap=2, sharey=False, aspect=1.4, legend_out=False)
    g.map_dataframe(sns.histplot, x=feature, bins=20, kde=True, stat="density", label="KDE")
    g.set_titles(col_template="{col_name}")
    for col_val, ax in g.axes_dict.items():
        cur_feat = all_df[all_df["playlist"] == col_val][feature]
        dist = fit_beta(cur_feat)
        dists.append(dist)
        x = np.linspace(0.001, 0.999, 100)
        ax.plot(x, dist.pdf(x), "r-", lw=2, alpha=0.6, label="beta pdf")
        ax.set_xlim([0, 1])
        ax.legend()
    g.fig.suptitle(f"Distribution of {feature} for different playlists", y=1.05)
    plt.show()
    return dists

def plot_feature(feature: str, nbins: int = 20) -> None:
    g1 = sns.FacetGrid(all_df, col="playlist", col_wrap=2, sharey=False, aspect=1.4)
    g1.map_dataframe(sns.histplot, x=feature, bins=nbins, kde=True, stat="density", label="KDE")
    g1.set_titles(col_template="{col_name}")
    for col_val, ax in g1.axes_dict.items():
        mean = all_df[all_df["playlist"] == col_val][feature].mean()
        ax.axvline(mean, ls="--", c="r", label="mean")
        ax.legend()
    g1.fig.suptitle(f"Distribution of {feature} for different playlists", y=1.05)

    g2 = sns.displot(data=all_df, x=feature, hue="playlist", kind="kde")
    g2.set(title=f"KDE for {feature} across different playlists")

    plt.show()


def fit_and_plot_beta(feature: str, fit_dist, nbins: int = 20) -> list:
    dists = []
    g = sns.FacetGrid(all_df, col="playlist", col_wrap=2, sharey=False, aspect=1.4, legend_out=False)
    g.map_dataframe(sns.histplot, x=feature, bins=nbins, kde=True, stat="density", label="KDE")
    g.set_titles(col_template="{col_name}")
    for col_val, ax in g.axes_dict.items():
        cur_feat = all_df[all_df["playlist"] == col_val][feature]
        dist = fit_dist(cur_feat)
        dists.append(dist)
        x = np.linspace(0.001, 0.999, 100)
        ax.plot(x, dist.pdf(x), "r-", lw=2, alpha=0.6, label="fit pdf")
        ax.set_xlim([0, 1])
        ax.legend()
    g.fig.suptitle(f"Distribution of {feature} for different playlists", y=1.05)
    plt.show()
    return dists
