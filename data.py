import numpy as np
import itertools
from scipy.optimize import linear_sum_assignment


class Distribution():
    def __init__(self):
        self.n = None
        self.t_len = None
        self.trails = None
        self.trail_probs = None
        self.all_trail_probs_ = None

    def from_mixture(mixture, t_len):
        self = Distribution()
        self.n = mixture.n
        self.t_len = t_len
        self.trails = np.array(list(itertools.product(range(self.n), repeat=t_len)))
        self.all_trail_probs_ = np.zeros([self.n] * t_len)

        Ms_log = np.log(1e-16 + mixture.Ms.astype(np.longdouble))
        S_log = np.log(1e-16 + mixture.S.astype(np.longdouble))
        for l in range(mixture.L):
            for trail in self.trails:
                trail_prob_log = np.array(S_log[l,trail[0]], dtype=np.longdouble)
                for i, j in zip(trail, trail[1:]):
                    trail_prob_log += Ms_log[l, i, j]
                self.all_trail_probs_[tuple(trail)] += np.exp(trail_prob_log)

        assert(np.all(np.isfinite(self.all_trail_probs_)))
        self.trail_probs = self.all_trail_probs_.flatten()
        return self

    def from_all_trail_probs(all_trail_probs):
        self = Distribution()
        self.n = len(all_trail_probs)
        self.t_len = all_trail_probs.ndim
        self.trails = np.array(list(itertools.product(range(self.n), repeat=self.t_len)))
        self.all_trail_probs_ = all_trail_probs
        self.trail_probs = all_trail_probs.flatten()
        return self

    def compressed_trails(self):
        ix_sort = np.lexsort(np.rot90(self.trails))
        sorted_trails = self.trails[ix_sort]
        sorted_trail_probs = self.trail_probs[ix_sort]
        trails, ix_start, count = np.unique(sorted_trails, return_counts=True, return_index=True, axis=0)
        trail_probs = np.array([ np.sum(sorted_trail_probs[s:s+c]) for s, c in zip(ix_start, count) ])
        return trails, trail_probs

    def restrict_to(self, states):
        tp = self.all_trail_probs()[states][:,states][:,:,states]
        return Distribution.from_all_trail_probs(tp / np.sum(tp))

    def compress_trail(self):
        self.trails, self.trail_probs = self.compressed_trails()

    def all_trail_probs(self):
        if self.all_trail_probs_ is not None:
            return self.all_trail_probs_
        trails, trail_probs = self.compressed_trails()
        self.all_trail_probs_ = np.zeros([self.n] * self.t_len)
        for trail, trail_prob in zip(trails, trail_probs):
            self.all_trail_probs_[tuple(trail)] = trail_prob
        return self.all_trail_probs_

    def from_trails(n, trails):
        self = Distribution()
        self.n = n
        self.t_len = len(trails[0])
        self.trails = np.array(trails)
        num_trails = len(trails)
        self.trail_probs = np.ones(num_trails) / num_trails
        self.all_trail_probs_ = None
        return self

    def sample(self, n_samples=1, eps=None, exponential=False, gaussian=False):
        sample = Distribution()
        sample.n = self.n
        sample.t_len = self.t_len
        sample.trails = self.trails

        if eps is None:
            d = int(1e5)
            sample_trail_counts = np.zeros(len(self.trails))
            for k_samples in (n_samples // d) * [d] + [n_samples % d]:
                sample_ixs = np.random.choice(range(len(self.trails)), size=k_samples, p=self.trail_probs)
                values, counts = np.unique(sample_ixs, return_counts=True)
                sample_trail_counts[values] += counts
            sample.trail_probs = sample_trail_counts / n_samples
        else:
            flat_probs = self.all_trail_probs().flatten()
            sample.trails = np.array(list(itertools.product(range(self.n), repeat=self.t_len)))
            if exponential: noise = np.random.exponential(eps, self.n ** self.t_len)
            elif gaussian: noise = np.random.normal(0, eps, self.n ** self.t_len)
            else: noise = eps * (2 * np.random.rand(self.n ** self.t_len) - 1)
            sample_trail_probs = np.abs(flat_probs + noise)
            sample.trail_probs = sample_trail_probs / np.sum(sample_trail_probs)

        return sample

    def dist(self, other_distribution):
        return np.sum(np.abs(self.all_trail_probs() - other_distribution.all_trail_probs())) / 2

    def flat_trails(self):
        flat_trails = np.zeros((len(self.trails), self.n + self.n**2))
        for trail_n, trail in enumerate(self.trails):
            flat_trails[trail_n, trail[0]] = 1
            for i, j in zip(trail, trail[1:]):
                flat_trails[trail_n, self.n + self.n*i + j] += 1
        return flat_trails, self.trail_probs

    def combine_uniform(self, L, a=1/4):
        assert(self.t_len == 3)

        comb = Distribution()
        comb.n = self.n
        comb.t_len = 3
        comb.trails = np.array(list(itertools.product(range(self.n), repeat=3)))
        comb.trail_probs = np.empty(self.n**3)

        all_trail_probs = self.all_trail_probs()
        trail2_probs = np.sum(all_trail_probs, axis=2)
        for t, (i, j, k) in enumerate(comb.trails):
            comb.trail_probs[t] = \
                (1-a)**2 * all_trail_probs[i,j,k] + a**2 / self.n**3 +\
                a*(1-a) * trail2_probs[i,j] / self.n + a*(1-a) * trail2_probs[j, k] / self.n

        return comb

    """
    def sample_trails(self, n_samples=1):
        trails = np.array(list(itertools.product(range(self.mixture.n), repeat=self.t_len)))
        sample_ixs = np.random.choice(range(self.num_trails), size=n_samples, p=self.trail_probs_flat)
        return trails[sample_ixs]

    def sample_probs(self, n_samples=1):
        nums = np.arange(self.num_trails).reshape([self.mixture.n] * self.t_len)
        dists = np.array([nums == x for x in range(self.num_trails)])
        sample_ixs = np.random.choice(range(self.num_trails), size=n_samples, p=self.trail_probs_flat)
        return np.sum(dists[sample_ixs], axis=0) / n_samples

    def sample_artificial_probs(self, eps=0.01):
        return self.trail_probs + eps * (2 * np.random.rand(*([self.mixture.n] * self.t_len)) - 1)
    """


"""
class Sample():
    def __init__(self, mixture, t_len, n_samples=1):
        self.mixture = mixture
        self.t_len = t_len
        self.n_samples = n_samples

        num_ts = mixture.n ** t_len
        num_samples = mixture.L * num_ts
        self.direct_all = np.empty((num_samples, t_len), dtype=np.int8)
        self.vectors_all = np.zeros((num_samples, mixture.n * t_len), dtype=np.int8)
        self.distr_all = np.empty((num_samples, mixture.n ** t_len), dtype=np.int8)
#       self.transition_all = ...
        self.prob_all = np.empty(mixture.L * mixture.n ** t_len)

        Ms_log = np.log(mixture.Ms.astype(np.longdouble))
        n_sample = 0
        for l in range(mixture.L):
            self.distr_all[l*num_ts:(l+1)*num_ts, :] = np.eye(num_ts)
            for ts in itertools.product(range(mixture.n), repeat=t_len):
                prob = np.array(np.log(mixture.S[l, ts[0]]), dtype=np.longdouble)
                for i, j in zip(ts, ts[1:]):
                    prob += Ms_log[l,i,j]
                self.prob_all[n_sample] = np.exp(prob)

                self.direct_all[n_sample] = ts
                self.vectors_all[n_sample, mixture.n * np.arange(t_len) + ts] = 1
                n_sample += 1

        self.sample_ixs = np.random.choice(range(num_samples), size=n_samples,
                p=(self.prob_all / np.sum(self.prob_all)).astype(np.float64))

    def direct(self):
        return self.direct_all[self.sample_ixs]

    def vectors(self):
        return self.vectors_all[self.sample_ixs]

    def distr(self):
        T = self.distr_all[self.sample_ixs].reshape([self.n_samples] + [self.mixture.n] * self.t_len)
        return T.sum(axis=0) / self.n_samples

    def transition(self):
        pass
"""


class Mixture():
    def __init__(self, S, Ms):
        self.L, self.n = S.shape
        assert(Ms.shape == (self.L, self.n, self.n))
        self.S = S
        self.Ms = Ms

    def normalize(self):
        self.S /= np.sum(self.S)
        self.Ms = self.Ms / np.sum(self.Ms, axis=2)[:, :, np.newaxis]
        self.Ms[np.any(np.isnan(self.Ms), axis=2)] = 1 / self.n

    def from_flat(flat_mixture, n):
        S = flat_mixture[:, :n]
        Ms = np.array([ row[n:].reshape((n,n)) for row in flat_mixture ])
        return Mixture(S, Ms)

    def random(n, L, seed=None):
        rng = np.random.default_rng(seed)
        S = rng.random((L, n))
        Ms = rng.random((L, n, n))
        mixture = Mixture(S, Ms)
        mixture.normalize()
        return mixture

    def perm_dist(A, B):
        l = len(A)
        assert(l == len(B))
        d = np.array([[np.sum(np.abs(A[l1] - B[l2])) for l2 in range(l)] for l1 in range(l)])
        row_ind, col_ind = linear_sum_assignment(d)
        return np.sum(d[row_ind, col_ind]) / (2 * l)

    def recovery_error(self, other_mixture):
        if not (self.L == other_mixture.L and self.n == other_mixture.n): return np.inf
        # assert(self.L == other_mixture.L and self.n == other_mixture.n)
        d = np.zeros((self.L, self.L))
        for l1 in range(self.L):
            for l2 in range(self.L):
                d[l1, l2] = np.sum(np.abs(self.Ms[l1] - other_mixture.Ms[l2])) / (2*self.n)
                # what about S?
        row_ind, col_ind = linear_sum_assignment(d)
        # return Mixture.perm_dist(self.Ms, other_mixture.Ms) / self.n
        return d[row_ind, col_ind].sum() / self.L

    def print(self):
        with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
            print("------- S:")
            print(self.S)
            for i, M in enumerate(self.Ms):
                print("------- M{}:".format(i))
                print(self.Ms[i])

    def __repr__(self):
        return f"Mixture(n={self.n}, L={self.L})"

    def __str__(self):
        with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
            x = "Mixture(\n "
            x += str(self.S).replace('\n', '\n ')
            x += "\n,\n "
            x += str(self.Ms).replace('\n', '\n ')
            x += "\n)"
            return x

    def flat(self):
        return np.array([np.hstack((s, M.flatten())) for s, M in zip(self.S, self.Ms)])

    def combine_uniform(self, a):
        return Mixture(self.S, (1-a) * self.Ms + a * np.ones(self.Ms.shape) / self.n)

    def restrict_to(self, states):
        m = Mixture(self.S[:, states], self.Ms[:,states][:,:,states])
        m.normalize()
        return m

    def copy(self):
        return Mixture(self.S.copy(), self.Ms.copy())



"""
def rnd_stoch_matrix(n):
    M = np.random.rand(n, n)
    return M / M.sum(axis=1)[:,None]

def rnd_mixture(n, L):
    S = np.random.rand(L, n)
    S /= np.sum(S)
    return (S, [ rnd_stoch_matrix(n) for l in range(L) ])

def sample_mixture(mixture, t_len=3, n_samples=1):
    S, Ms = mixture
    L, n = S.shape

    samples = np.empty((n_samples, t_len), dtype=np.int8)

    for i in range(n_samples):
        x = np.random.choice(np.arange(L*n), p=S.flatten())
        M = Ms[x // n]
        state = x % n
        samples[i, 0] = state
        for t in range(1, t_len):
            state = np.random.choice(np.arange(n), p=M[state,:])
            samples[i, t] = state

    return samples


def efficient_sample(mixture, t_len=3, n_samples=1):
    S, Ms = mixture
    Ms_log = [ np.log(M.astype(np.longdouble)) for M in Ms ]
    L, n = S.shape
    num_ts = L * n**t_len
    ts_all = 99 * np.ones((num_ts, t_len), dtype=np.int8)
    ts_all_prob = np.empty(num_ts, dtype=np.longdouble)

    n_sample = 0
    for l in range(L):
        for ts in itertools.product(range(n), repeat=t_len):
            prob = np.array(S[l, ts[0]], dtype=np.longdouble)
            for i, j in zip(ts, ts[1:]):
                prob += Ms_log[l][i,j]
            ts_all[n_sample] = ts
            ts_all_prob[n_sample] = np.exp(prob)
            n_sample += 1

    sample_ixs = np.random.choice(range(num_ts), size=n_samples,
            p=(ts_all_prob / np.sum(ts_all_prob)).astype(np.float64))
    return ts_all[sample_ixs]
"""


# mixture = rnd_mixture(3, 5)

# print(sample_mixture(mixture, n_samples=10))


# mixture = Mixture.random(3, 2)
# d = Distribution(mixture, 3)
# print(d.sample_direct(1000))
# sample = Sample(mixture, 3, n_samples=1000)
# print(sample.vectors()[10])


