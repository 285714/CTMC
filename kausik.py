from data import *
import itertools
# import sklearn
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt


def sample_trails(mixture, n_samples, t_len):
    trails = np.zeros((n_samples, t_len), dtype=int)
    flatS = mixture.S.flatten()
    starting = np.random.choice(range(len(flatS)), size=n_samples, p=flatS)
    trails[:,0], ls = np.divmod(starting, mixture.L)

    rnd = np.random.random_sample((n_samples, t_len-1))
    cum = np.cumsum(mixture.Ms, axis=2)
    for i in range(1, t_len):
        intvals = cum[ls, trails[:,i-1]]
        trails[:,i] = np.sum(rnd[:,i-1][:,None] > intvals, axis=1)

    # print(ls)
    return trails


def kausik_learn(n, L, trails):
    n_samples, t_len = trails.shape
    segment_len = t_len // 4


    # subspace estimation
    M = np.zeros((n, n, n))
    next_state = np.zeros((n, n_samples, 2, n))
    for i in range(n):
        for t, trail in enumerate(trails):
            for segno, trail_segment in enumerate([trail[segment_len:2*segment_len], trail[3*segment_len:4*segment_len]]):
                states, counts = np.unique(trail_segment[1:][trail_segment[:-1] == i], return_counts=True)
                next_state[i, t, segno,states] = counts / np.sum(counts)
            M[i] += np.outer(next_state[i, t, 0], next_state[i, t, 1])
    M /= n_samples
    X = M + np.moveaxis(M, 1, 2)
    u, s, vh = np.linalg.svd(X)
    V = np.zeros((n, n, n))
    for i in range(n):
        V[i] = u[i,:,:L] @ np.diag(s[i,:L]) @ vh[i,:L,:]

    # clustering
    dist = np.zeros((n_samples, n_samples))
    for t1, t2 in itertools.product(range(n_samples), repeat=2):
        D = np.zeros((2, n, n))
        for i in range(n):
            for segno in range(2):
                D[segno, i] = V[i] @ (next_state[i, t1, segno] - next_state[i, t2, segno])
        dist[t1, t2] = np.max(np.inner(D[0], D[1]))
    clustering = AgglomerativeClustering(metric="precomputed", linkage="complete", n_clusters=L).fit(dist)
    labels = clustering.labels_
    # print(labels)

    # learning
    S = np.zeros((L, n))
    Ms = np.zeros((L, n, n))
    for l in range(L):
        chain_trails = trails[labels == l]
        states, counts = np.unique(chain_trails[:, 0], return_counts=True)
        S[l, states] = counts / np.sum(counts)
        for i in range(n):
            states, counts = np.unique(chain_trails[:,1:][chain_trails[:,:-1] == i], return_counts=True)
            Ms[l,i,states] = counts / np.sum(counts)

    return Mixture(S, Ms)


def em_long_trails(n, L, trails, n_iter=100):
    n_samples, t_len = trails.shape
    mixture = Mixture.random(n, L)
    eps = 1e-10

    for _ in range(n_iter):
        logS = np.log(mixture.S + eps)
        logMs = np.log(mixture.Ms)

        logl = logS[:, trails[:,0]] # L x n_samples
        for i in range(1, t_len):
            logl += logMs[:, trails[:,i-1], trails[:,i]]
        probs = np.exp(logl - np.max(logl, axis=0))
        probs /= np.sum(probs, axis=0)[None, :]

        for l in range(L):
            mixture.S[l] = probs[l] @ (trails[:,0][:,None] == np.arange(n)[None,:])
            next_state = probs[l][:,None,None] * (trails[:,1:][:,:,None] == np.arange(n)[None,None,:])
            for i in range(n):
                X = (trails[:,:-1] == i)[:,:,None] * next_state
                mixture.Ms[l,i,:] = np.sum(X, axis=(0,1))
        mixture.S += eps
        mixture.Ms += eps
        mixture.normalize()

    return mixture


def learn(n, L, trails, method):
    if method == "kausik":
        return kausik_learn(n, L, trails)
    elif method == "em":
        return em_long_trails(n, L, trails)
    else:
        assert(False)



df = pd.DataFrame(itertools.product([5], [2], np.linspace(10, 100, 10, dtype=int), [10000]), columns=["n", "L", "n_samples", "t_len"])
df["mixture"] = df.apply(lambda x: Mixture.random(x.n, x.L), axis=1)
df = df.merge(pd.DataFrame(range(5), columns=["n_trial"]), how="cross")
df["sample_trails"] = df.apply(lambda x: sample_trails(x.mixture, x.n_samples, t_len=x.t_len), axis=1)
df = df.merge(pd.DataFrame(["kausik", "em"], columns=["method"]), how="cross")
df["learned_mixture"] = df.apply(lambda x: learn(x.n, x.L, x.sample_trails, x.method), axis=1)
df["recovery_error"] = df.apply(lambda x: x.mixture.recovery_error(x.learned_mixture), axis=1)

for method, df2 in df.groupby("method"):
    grp = df2.groupby("n_samples")
    mean = grp["recovery_error"].mean()
    std = grp["recovery_error"].std()
    ax = mean.plot(label=method)
    ax.fill_between(std.index, mean-std, mean+std, alpha=0.2)

ax.legend()
ax.set_ylabel("recovery_error")
ax.set_title(f"n={df.iloc[0].n}, L={df.iloc[0].L}, t_len={df.iloc[0].t_len}")

plt.tight_layout()
plt.savefig("test1.pdf")


"""
df = pd.DataFrame(itertools.product([5], [2], [50], np.linspace(1000, 100000, 10, dtype=int)), columns=["n", "L", "n_samples", "t_len"])
df["mixture"] = df.apply(lambda x: Mixture.random(x.n, x.L), axis=1)
df = df.merge(pd.DataFrame(range(5), columns=["n_trial"]), how="cross")
df["sample_trails"] = df.apply(lambda x: sample_trails(x.mixture, x.n_samples, t_len=x.t_len), axis=1)
df = df.merge(pd.DataFrame(["kausik", "em"], columns=["method"]), how="cross")
df["learned_mixture"] = df.apply(lambda x: learn(x.n, x.L, x.sample_trails, x.method), axis=1)
df["recovery_error"] = df.apply(lambda x: x.mixture.recovery_error(x.learned_mixture), axis=1)

for method, df2 in df.groupby("method"):
    grp = df2.groupby("n_samples")
    mean = grp["recovery_error"].mean()
    std = grp["recovery_error"].std()
    ax = mean.plot(label=method)
    ax.fill_between(std.index, mean-std, mean+std, alpha=0.2)

ax.legend()
ax.set_ylabel("recovery_error")
ax.set_title(f"n={df.iloc[0].n}, L={df.iloc[0].L}, n_samples={df.iloc[0].n_samples}")

plt.tight_layout()
plt.savefig("test2.pdf")
"""













"""
df = pd.DataFrame(itertools.product([5], [2], [50], np.linspace(1000, 100000, 20, dtype=int)), columns=["n", "L", "n_samples", "t_len"])
df["mixture"] = df.apply(lambda x: Mixture.random(x.n, x.L), axis=1)
df = df.merge(pd.DataFrame(range(5), columns=["n_trial"]), how="cross")
df["sample_trails"] = df.apply(lambda x: sample_trails(x.mixture, x.n_samples, t_len=x.t_len), axis=1)
df["learned_mixture"] = df.apply(lambda x: kausik_learn(x.n, x.L, x.sample_trails), axis=1)
df["recovery_error"] = df.apply(lambda x: x.mixture.recovery_error(x.learned_mixture), axis=1)

grp = df.groupby("t_len")
mean = grp["recovery_error"].mean()
std = grp["recovery_error"].std()
ax = mean.plot()
ax.fill_between(std.index, mean-std, mean+std, alpha=0.2)
ax.set_ylabel("recovery_error")
ax.set_title(f"n={df.iloc[0].n}, L={df.iloc[0].L}, n_samples={df.iloc[0].n_samples}")

plt.tight_layout()
plt.savefig("test2.pdf")
"""

"""
n = 5
L = 2

mixture = Mixture.random(n, L)
# print(mixture)

trails = sample_trails(mixture, 100, t_len=100)
learned_mixture = kausik_learn(n, L, trails)
print("kausik", mixture.recovery_error(learned_mixture))

learned_mixture2 = em_long_trails(n, L, trails)
# print(learned_mixture2)
print("    em", mixture.recovery_error(learned_mixture2))
"""


