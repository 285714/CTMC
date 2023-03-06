import itertools
import warnings
from data import *
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.vq import kmeans2, ClusterError
import time
import cvxpy as cp


def svd_multi_sample_dist(sample, *args, repeat=5, **kwargs):
    best_learned_mixture = None
    best_dist = np.inf
    exc = None
    for sample_dist in np.linspace(0.01, 1, 5):

        try:
            learned_mixture = repeated(svd_learn_new, sample, *args, sample_dist=sample_dist, max_reps=repeat, **kwargs)
            learned_distribution = Distribution.from_mixture(learned_mixture, 3)
            d = learned_distribution.dist(sample)
            if d < best_dist:
                best_dist = d
                best_learned_mixture = learned_mixture

        except Exception as e:
            exc = e
            import traceback
            print(traceback.format_exc())

    if best_learned_mixture is None:
        raise exc

    return best_learned_mixture


def repeated(f, sample, *args, max_reps=10, target_dist=0.01, **kwargs):
    best_learned_mixture = None
    best_dist = np.inf
    for i in range(max_reps):
        learned_mixture = f(sample, *args, **kwargs)
        learned_distribution = Distribution.from_mixture(learned_mixture, 3)
        d = learned_distribution.dist(sample)
        if d < best_dist:
            best_dist = d
            best_learned_mixture = learned_mixture
        if best_dist < target_dist:
            break
    return best_learned_mixture


class SVDLearn():
    def __init__(self,
                 group_min_dist=1,
                 group_num=None,
                 ingroup_n_pairs=1,
                 ingroup_compute=None,
                 ingroup_combine=None,
                 ingroup_normalization=None,
                 ingroup_use_robust_mean=True,
                 combine=None,
                 use_robust_mean=True,
                 compress=True,
                 normalization=None,
                 verbose=False):
        self.group_min_dist = group_min_dist
        self.group_num = group_num
        self.ingroup_n_pairs = ingroup_n_pairs
        self.ingroup_compute = ingroup_compute or SVDLearn.ingroup_compute_direct
        self.ingroup_combine = ingroup_combine or SVDLearn.ingroup_combine_take_first
        self.ingroup_normalization = ingroup_normalization or SVDLearn.normalization_none
        self.ingroup_use_robust_mean = ingroup_use_robust_mean
        self.combine = combine or SVDLearn.combine_R
        self.use_robust_mean = use_robust_mean
        self.compress = compress
        self.normalization = normalization or SVDLearn.normalization_abs
        self.verbose = verbose

    class Guess():
        def __init__(self, L, r, Ps, Qs, Ys, Zs):
            self.L = L
            self.r = r
            self.Ps = Ps
            self.Qs = Qs
            self.Ys = Ys
            self.Zs = Zs
            self.prods = Zs @ np.transpose(Ys, axes=(0,2,1))
            self.prods_inv = np.linalg.pinv(self.prods)

    def learn(self, sample, L=None):
        Os = np.moveaxis(sample.all_trail_probs(), 1, 0)
        guess = self.guess_decomp(L, sample.n, Os)

        dists = self.state_dists(guess.r, guess)
        dist_mtrx = squareform(dists)
        dist_mtrx = dist_mtrx + np.tril(np.inf * np.ones(dist_mtrx.shape))
        groups = self.group(dists, group_min_dist=self.group_min_dist, group_num=self.group_num)

        xs = []
        for g in range(max(groups) + 1):
            (states,) = np.where(groups == g)
            mixture, R = self.ingroup_compute(self, guess, Os, dist_mtrx, states)
            mixture, R = self.ingroup_normalization(self, mixture, R, sample, states)
            xs.append((mixture, R))

        mixture = self.combine(self, xs, guess)
        mixture = self.normalization(self, mixture, None, sample, None)
        return mixture

    def ingroup_compute_direct(self, guess, Os, dist_mtrx, states):
        s1, s2 = np.array_split(states, 2) # could also pair them all up
        assert(len(s1) > 0 and len(s2) > 0)
        R = self.reconstruct_R(guess, Os, s1, s2)
        mixture = self.reconstruct(R, guess)
        return mixture, R

    def ingroup_compute_on_pairs(self, guess, Os, dist_mtrx, states):
        pairs = self.find_closest_pairs(dist_mtrx, states)
        Rs = [SVDLearn.reconstruct_R(self, guess, Os, [i], [j]) for i, j in pairs]
        mixture, R = self.ingroup_combine(self, Rs, states, guess)
        return mixture, R

    def normalization_none(self, mixture, R, sample, states):
        return mixture, R

    def normalization_abs(self, mixture, R, sample, states):
        mixture = Mixture(np.abs(mixture.S), np.abs(mixture.Ms))
        mixture.normalize()
        return mixture

    def em_refine(self, mixture, sample, states):
        if states is not None:
            states = list(states)
            all_trail_probs = sample.all_trail_probs()
            states_trail_probs = all_trail_probs[states][:,states][:,:,states]
            states_trail_probs /= np.sum(states_trail_probs)
            d = Distribution.from_all_trail_probs(states_trail_probs)
        else:
            d = sample
        return em_learn(d, sample.n, mixture.L, max_iter=10, init_mixture=mixture)

    def normalization_abs_em(self, mixture, R, sample, states):
        mixture = self.normalization_abs(mixture, None, sample, states)
        return self.em_refine(mixture, sample, states)

    def ingroup_combine_median_R(self, Rs, guess):
        mixture = self.ingroup_combine_median(self, Rs, guess)
        assert(False)

    def find_representative(X):
        X = np.array(X)
        Y = np.empty(X.shape)
        Y[:] = np.nan
        ixs = (X > -0.5) & (X < 1.5)
        Y[ixs] = X[ixs]
        return np.nanmedian(Y, axis=0)

    def ingroup_combine_median(self, Rs, states, guess):
        all_S, all_Ms = [], []
        tvdists = []
        for R in Rs:
            mixture = self.reconstruct(self, R, guess)
            all_S.append(mixture.S[:, states])
            all_Ms.append(mixture.Ms[:, states])
        S = self.find_representative(all_S)
        Ms = self.find_representative(all_Ms)
        return Mixture(S, Ms), None

    def ingroup_combine_take_first(self, Rs, states, guess):
        mixture = self.reconstruct(Rs[0], guess)
        return mixture, Rs[0]

    def combine_mixture(mixtures, guess):
        pass

    def combine_take_first(self, xs, guess):
        mixture, R = xs[0]
        return mixture or self.reconstruct(R, guess)

    def combine_R(self, Rs, guess):
        Rs_inv = [np.linalg.pinv(R) for _, R in Rs]
        collectedRinv = np.vstack(list(R_inv.T for R_inv in Rs_inv))
        collectedRinvOrigin = [ i for i, R_inv in enumerate(Rs_inv) for _ in range(len(R_inv.T)) ]

        dists = []
        for (j1, v1), (j2, v2) in itertools.combinations(zip(collectedRinvOrigin, collectedRinv), 2):
            dists.append(9e99 if j1 == j2 else min(self.vec_dist(v1, v2), self.vec_dist(v1, -v2)))
        groups = self.group(dists, group_num=guess.r)

        combinedRinv = np.zeros((guess.r, guess.r))
        for g in range(max(groups) + 1):
            vs = collectedRinv[np.where(groups == g)]
            repr_v = self.robust_mean(vs) if self.use_robust_mean else np.mean(vs, axis=0)
            combinedRinv[g] = repr_v

        R = np.linalg.pinv(combinedRinv.T)
        Ys = R @ guess.Ys
        comp = self.asgn_mtrx(guess, np.linalg.norm(Ys, axis=2, ord=1).T) if self.compress else np.eye(guess.r)
        compressedR = comp @ R

        return self.reconstruct(compressedR, guess)

    def asgn_mtrx(self, guess, mass):
        A = cp.Variable((guess.L, guess.r), boolean=True)
        objective = cp.Minimize(cp.sum(cp.max(A @ mass, axis=0)))
        constraint = cp.sum(A, axis=0) == 1
        prob = cp.Problem(objective, [constraint])
        try:
            prob.solve(verbose=False, solver="CBC", maximumSeconds=5)
            assert(A.value.shape == (guess.L, guess.r))
            return A.value
        except Exception as e:
            print("solver exception:", e)
            return np.tile(np.eye(guess.L), guess.r // guess.L + 1)[:,:guess.r]

    def combine_R___(Rs, guess):
        # TODO
        Rs_inv = [np.linalg.pinv(R) for R in Rs]
        collectedRinv = np.real(np.vstack(list(p.Rinv.T for p in parts)))
        collectedRinvOrigin = [ i for i, p in enumerate(parts) for _ in range(len(p.Rinv.T)) ]

        # dists = pdist(collectedRinv / np.linalg.norm(collectedRinv, axis=0))
        dists = np.zeros(len(collectedRinv) * (len(collectedRinv) - 1) // 2)
        for k, ((p1, rinv1), (p2, rinv2)) in enumerate(itertools.combinations(zip(collectedRinvOrigin, collectedRinv), 2)):
            dists[k] = np.linalg.norm(rinv1 - rinv2)**2 / (np.linalg.norm(rinv1) * np.linalg.norm(rinv2))
            if p1 == p2:
                dists[k] = 10
            elif parts[p1].i == parts[p2].j or parts[p1].j == parts[p2].i:
                dists[k] /= 10

        # fcluster seems buggy, so here's a quick fix
        dist_mtrx = squareform(dists + 1e-10 * np.random.rand(*dists.shape))
        double_dists = [0 if i//2 == j//2 else dist_mtrx[i//2, j//2]
                        for i, j in itertools.combinations(range(2 * len(dist_mtrx)), 2)]
        lnk = linkage(double_dists, method="complete")
        double_groups = fcluster(lnk, r, criterion="maxclust") - 1
        groups = np.array([g for i, g in enumerate(double_groups) if i%2])
        assert(max(groups)+1 == r)

        combinedRinv = np.zeros((r, r))
        for l in range(r):
            cluster = collectedRinv[groups==l]
            intra_dists = np.sum(squareform(dists)[groups==l][:,groups==l], axis=0)
            center = cluster[np.argmin(intra_dists)]
            combinedRinv[l] = center
            if verbose:
                with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
                    avg_dist = np.average(
                        list(np.linalg.norm(x - y, ord=1) for x, y in itertools.combinations(cluster, 2))) if len(
                        cluster) > 1 else 0
                    cen_dist = np.linalg.norm(cluster - center, ord=1, axis=1)
                    print("-" * 10,
                          f"label={l} (size={len(cluster)}, d={avg_dist:.5f}) dist from center: avg={np.average(cen_dist):.5f} max={np.max(cen_dist):.5f}",
                          "-" * 10)
                    print("\n".join([
                        f"{'>' if np.allclose(collectedRinv[i], center) else ' '} {parts[i//L].i:2d} {parts[i//L].j:2d} ({i % L}) {x}"
                        for i, x in
                        zip(np.where(groups==l)[0], str(cluster).split("\n"))]))  # where(labels==l)

        assert(len(combinedRinv) == r)
        R = np.linalg.pinv(combinedRinv.T)

        def asgn_mtrx(mass):
            A = cp.Variable((L, r), boolean=True)
            objective = cp.Minimize(cp.sum(cp.max(A @ mass, axis=0)))
            constraint = cp.sum(A, axis=0) == 1
            prob = cp.Problem(objective, [constraint])
            try:
                # print("<", end="", flush=True)
                prob.solve(verbose=False, solver="CBC", maximumSeconds=5)
                # print(">", end="", flush=True)
                assert(A.value.shape == (L, r))
                return A.value
                # if problem.status == 'optimal': .... else: ....
            except Exception as e:
                print("solver exception:", e)
                return np.tile(np.eye(L), r // L + 1)[:,:r]

        Ys = R @ Ys_
        comp = asgn_mtrx(np.linalg.norm(Ys, axis=2, ord=1).T) if compress else np.eye(r)
        if verbose: print(comp)
        compressedR = comp @ R
        if verbose:
            print(f"compressedR.shape = {compressedR.shape} (ideal is {L},{r})")

        Ys = compressedR @ Ys_
        Zs = compressedR @ Zs_
        Ps = Ys @ Ps_
        S = np.real(np.diagonal(Zs @ np.transpose(Ys, axes=(0,2,1)), axis1=1, axis2=2).T)
        Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
        return S, Ms

    def reconstruct(self, R, guess):
        Ys = R @ guess.Ys
        Ps = Ys @ guess.Ps
        S = np.real(np.diagonal(R @ guess.Zs @ np.transpose(guess.Ys, axes=(0, 2, 1)) @ R.T, axis1=1, axis2=2).T)
        Ms = np.real(np.transpose(Ps / S, axes=(1, 2, 0)))
        return Mixture(S, Ms)

    def reconstruct_R(self, guess, Os, s1, s2):
        Rs = []
        for i, j in itertools.combinations(list(s1) + list(s2), 2): # zip(s1, s2):
            E = np.linalg.lstsq(guess.prods[i], guess.prods[j], rcond=None)[0]
            eigs, w = np.linalg.eig(E)
            mask = np.argpartition(eigs, -guess.L)[-guess.L:]
            R_ = w[:, mask]
            d = np.linalg.lstsq((R_.T @ guess.Ys[i] @ guess.Ps[i]).T, np.sum(Os[i], axis=1), rcond=None)[0]
            R = R_.T * d[:,None]
            Rs.append(R)

        dists = []
        for (i1, v1), (i2, v2) in itertools.combinations(enumerate(np.vstack(Rs)), 2):
            j1, j2 = i1 // guess.L, i2 // guess.L
            dists.append(9e99 if j1 == j2 else min(self.vec_dist(v1, v2), self.vec_dist(v1, -v2)))
        # groups = self.group(dists, group_num=guess.L)

        lnk = linkage(np.array(dists), method="ward")
        groups = fcluster(lnk, guess.L, criterion="maxclust") - 1

        """
        from scipy.cluster.hierarchy import dendrogram
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        plt.figure(1, figsize=(15, 10))
        plt.subplot(211)
        dendrogram(lnk)
        tsne = TSNE(n_components=2, perplexity=4, metric="precomputed")
        tsne_results = tsne.fit_transform(squareform(dists))
        plt.subplot(212)
        for l in range(max(groups) + 1):
            plt.scatter(tsne_results[groups == l, 0], tsne_results[groups == l, 1], s=100)
        plt.show()
        """

        R = []
        for g in range(max(groups) + 1):
            vs = np.vstack(Rs)[np.where(groups == g)]
            repr_v = self.robust_mean(vs) if self.ingroup_use_robust_mean else np.mean(vs, axis=0)
            R.append(repr_v)

        if self.verbose:
            for g in range(max(groups) + 1):
                print("-" * 100)
                vs = np.vstack(Rs)[np.where(groups == g)]
                print(f"group {g}: var={np.sum(np.var(vs, axis=0)):.5f}")
                with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
                    print(vs)
            print("-" * 100)
            print("result:")
            print(np.array(R))

        return np.array(R)

    def robust_mean(self, xs):
        xs = np.real(xs) # can I do this?
        np.random.shuffle(xs)
        r = 1
        while True:
            r *= 2
            xs_ = np.copy(xs) # or use zs
            ys, zs = np.array_split(xs, 2)
            for i, y in enumerate(ys.T):
                y = np.sort(y)
                k = len(y) // r
                _, u, l = min([ (u - l, u, l) for l, u in zip(y, y[-k:])])
                xs_ = xs_[np.where((u >= xs_[:,i]) & (xs_[:,i] > l))]
            if len(xs_) > 0 or r > 1000:
                return np.mean(xs_ if len(xs_) > 0 else xs, axis=0)



    """
    def reconstruct_R_old(n, L, guess, i, j, Os, use_all=True):
        if use_all:
            E = np.linalg.lstsq(np.vstack(guess.prods[n//2:]),
                                np.vstack(guess.prods[:n//2]), rcond=None)[0]
        else:
            E = guess.prods_inv[i] @ guess.prods[j]
        eigs, w = np.linalg.eig(E)
        mask = np.argpartition(eigs, -L)[-L:]
        R_ = w[:, mask]
        if use_all:
            d = np.linalg.lstsq((R_.T @ np.hstack(guess.Ys @ guess.Ps)).T, np.vstack(Os) @ np.ones(n), rcond=None)[0]
        else:
            d = np.linalg.lstsq((R_.T @ guess.Ys[i] @ guess.Ps[i]).T, Os[i] @ np.ones(n), rcond=None)[0]
        R = np.diag(d) @ R_.T
        return R
    """

    def find_closest_pairs(self, dist_mtrx, states):
        num_states = len(states)
        num_pairs = min(num_states * (num_states - 1) // 2, self.ingroup_n_pairs)
        group_dists = dist_mtrx[states]
        ixs = np.argpartition(group_dists, num_pairs, axis=None)[:num_pairs]
        pairs = np.unravel_index(ixs, group_dists.shape)
        return [(states[i], j) for i, j in zip(*pairs)]

    def group(self, dists, group_min_dist=1, group_num=None, method="complete"):
        if group_num is not None:
            dist_mtrx = squareform(dists)
            dists = [0 if i//2 == j//2 else dist_mtrx[i//2, j//2]
                     for i, j in itertools.combinations(range(2 * len(dist_mtrx)), 2)]
        lnk = linkage(np.array(dists), method=method)
        if group_num is None:
            groups = fcluster(lnk, group_min_dist, criterion="distance") - 1
        else:
            double_groups = fcluster(lnk, group_num, criterion="maxclust") - 1
            groups = np.array([g for i, g in enumerate(double_groups) if i%2])
            assert(max(groups) + 1 == group_num)
        return groups

    def vec_dist(self, x, y):
        return np.linalg.norm(x - y)**2 / (np.linalg.norm(x) * np.linalg.norm(y))

    def state_dists(self, r, guess):
        dists = []
        for (prod1, prod_inv1), (prod2, prod_inv2) in itertools.combinations(
                zip(guess.prods, guess.prods_inv), 2):
            E = prod_inv1 @ prod2
            E_inv = prod_inv2 @ prod1
            x = E @ np.random.rand(r, 50)
            y = E_inv @ E @ x
            dists.append(self.vec_dist(x, y))
        return dists

    def guess_decomp(self, L, n, Os):
        us, ss, vhs = np.linalg.svd(Os)
        if L is None:
            ss_norm = np.linalg.norm(ss, axis=0)
            ratios = ss_norm[:-1] / ss_norm[1:]
            L = 1 + np.argmax(ratios * (ss_norm[1:] > 1e-6))
        Ps_ = np.moveaxis(us, 1, 2)[:,:L]
        Qs_ = (ss[:,:L].T * vhs[:,:L].T).T

        A = np.zeros((2 * n * L, n ** 2))
        for j in range(n):
            A[L * j:L * (j + 1), n * j:n * (j + 1)] = Ps_[j]
            A[L * (n + j):L * (n + j + 1), j + n * (np.arange(n))] = -Qs_[j]

        _, s, vh = np.linalg.svd(A.T, full_matrices=True)
        s_inc = s[::-1]
        s_inc_c = s_inc[:n-1]
        ratios = s_inc_c[1:] / s_inc_c[:-1]
        r = max(L, 1 + np.argmax(ratios * (s_inc_c[1:] < 0.1)))
        if self.verbose:
            with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
                print("singular values of A:", " ".join( f"{x:.5f}" for x in s_inc))
                print("ratios:                      ", " ".join( f"{x:.5f}" for x in ratios))
                print(f"suggest r={r} (vs L={L})")
        B = np.random.rand(r,r) @ vh[-r:]
        Bre = np.moveaxis(B.reshape((r, L, 2 * n), order="F"), -1, 0)
        Ys_ = Bre[0:n]
        Zs_ = Bre[n:2*n]
        return self.Guess(L, r, Ps_, Qs_, Ys_, Zs_)

    def __call__(self, sample, *args, **kwargs):
        return self.learn(sample, *args, **kwargs)







def svd_learn_new(sample, n, L=None, compress=True, verbose=None, sample_dist=1, sample_num=None,
                  mixture=None, stats={}, distribution=None, sample_all=False, pair_selection=None,
                  em_refine_max_iter=0, returnsval=False):

    class PartialR:
        def __init__(self, R, states, i, j):
            self.R = R
            self.Rinv = np.linalg.pinv(R)
            self.states = set(states)
            self.i = i
            self.j = j

        def reconstruct(self):
            Ys = self.R @ Ys_
            Ps = Ys @ Ps_
            S = np.real(np.diagonal(self.R @ Zs_ @ np.transpose(Ys_, axes=(0,2,1)) @ self.R.T, axis1=1, axis2=2).T)
            Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
            return Mixture(S, Ms)

    def reconstruct_at(i, j):
        E = ProdsInv[i] @ Prods[j]
        eigs, w = np.linalg.eig(E)
        if L is not None:
            mask = np.argpartition(eigs, -L)[-L:]
        else:
            mask = eigs > 1e-5
        R_ = w[:, mask]
        d, _, _, _ = np.linalg.lstsq((R_.T @ Ys_[i] @ Ps_[i]).T, Os[i] @ np.ones(n), rcond=None)
        R = np.diag(d) @ R_.T
        return R

    def combine(parts):
        collectedRinv = np.real(np.vstack(list(p.Rinv.T for p in parts)))
        collectedRinvOrigin = [ i for i, p in enumerate(parts) for _ in range(len(p.Rinv.T)) ]

        # dists = pdist(collectedRinv / np.linalg.norm(collectedRinv, axis=0))
        dists = np.zeros(len(collectedRinv) * (len(collectedRinv) - 1) // 2)
        for k, ((p1, rinv1), (p2, rinv2)) in enumerate(itertools.combinations(zip(collectedRinvOrigin, collectedRinv), 2)):
            dists[k] = np.linalg.norm(rinv1 - rinv2)**2 / (np.linalg.norm(rinv1) * np.linalg.norm(rinv2))
            if p1 == p2:
                dists[k] = 10
            elif parts[p1].i == parts[p2].j or parts[p1].j == parts[p2].i:
                dists[k] /= 10

        # fcluster seems buggy, so here's a quick fix
        dist_mtrx = squareform(dists + 1e-10 * np.random.rand(*dists.shape))
        double_dists = [0 if i//2 == j//2 else dist_mtrx[i//2, j//2]
                        for i, j in itertools.combinations(range(2 * len(dist_mtrx)), 2)]
        lnk = linkage(double_dists, method="complete")
        double_groups = fcluster(lnk, r, criterion="maxclust") - 1
        groups = np.array([g for i, g in enumerate(double_groups) if i%2])
        assert(max(groups)+1 == r)

        combinedRinv = np.zeros((r, r))
        for l in range(r):
            cluster = collectedRinv[groups==l]
            intra_dists = np.sum(squareform(dists)[groups==l][:,groups==l], axis=0)
            center = cluster[np.argmin(intra_dists)]
            combinedRinv[l] = center
            if verbose:
                with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
                    avg_dist = np.average(
                        list(np.linalg.norm(x - y, ord=1) for x, y in itertools.combinations(cluster, 2))) if len(
                        cluster) > 1 else 0
                    cen_dist = np.linalg.norm(cluster - center, ord=1, axis=1)
                    print("-" * 10,
                          f"label={l} (size={len(cluster)}, d={avg_dist:.5f}) dist from center: avg={np.average(cen_dist):.5f} max={np.max(cen_dist):.5f}",
                          "-" * 10)
                    print("\n".join([
                        f"{'>' if np.allclose(collectedRinv[i], center) else ' '} {parts[i//L].i:2d} {parts[i//L].j:2d} ({i % L}) {x}"
                        for i, x in
                        zip(np.where(groups==l)[0], str(cluster).split("\n"))]))  # where(labels==l)

        assert(len(combinedRinv) == r)
        R = np.linalg.pinv(combinedRinv.T)

        def asgn_mtrx(mass):
            A = cp.Variable((L, r), boolean=True)
            objective = cp.Minimize(cp.sum(cp.max(A @ mass, axis=0)))
            constraint = cp.sum(A, axis=0) == 1
            prob = cp.Problem(objective, [constraint])
            try:
                # print("<", end="", flush=True)
                prob.solve(verbose=False, solver="CBC", maximumSeconds=5)
                # print(">", end="", flush=True)
                assert(A.value.shape == (L, r))
                return A.value
                # if problem.status == 'optimal': .... else: ....
            except Exception as e:
                print("solver exception:", e)
                return np.tile(np.eye(L), r // L + 1)[:,:r]

        Ys = R @ Ys_
        comp = asgn_mtrx(np.linalg.norm(Ys, axis=2, ord=1).T) if compress else np.eye(r)
        if verbose: print(comp)
        compressedR = comp @ R
        if verbose:
            print(f"compressedR.shape = {compressedR.shape} (ideal is {L},{r})")

        Ys = compressedR @ Ys_
        Zs = compressedR @ Zs_
        Ps = Ys @ Ps_
        S = np.real(np.diagonal(Zs @ np.transpose(Ys, axes=(0,2,1)), axis1=1, axis2=2).T)
        Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
        return S, Ms

    def em_refine(m, states=range(n), em_refine_max_iter=2):
        states = list(states)
        d = sample #.restrict_to(states)
        # m = mixture.restrict_to(states)
        return em_learn(d, len(states), L, max_iter=em_refine_max_iter, init_mixture=m)
        """
        all_trail_probs = sample.all_trail_probs()
        states_trail_probs = all_trail_probs[states][:,states][:,:,states]
        states_trail_probs /= np.sum(states_trail_probs)
        d = Distribution.from_all_trail_probs(states_trail_probs)
        return em_learn(d, len(states), L, max_iter=20, init_mixture=m)
        """

    def find_representative(X):
        X = np.array(X)
        Y = np.empty(X.shape)
        Y[:] = np.nan
        ixs = (X > -0.5) & (X < 1.5)
        Y[ixs] = X[ixs]
        return np.nanmedian(Y, axis=0)

    Os = np.moveaxis(sample.all_trail_probs(), 1, 0)
    us, ss, vhs = np.linalg.svd(Os)
    if L is None:
        ss_norm = np.linalg.norm(ss, axis=0)
        for i, s_norm in enumerate(ss_norm):
            stats[f"sval-{i}"] = s_norm
        ratios = ss_norm[:-1] / ss_norm[1:]
        for i, ratio in enumerate(ratios):
            stats[f"sval-ratio-{i}"] = ratio
        L = 1 + np.argmax(ratios * (ss_norm[:-1] > 1e-6))
        stats["guessedL"] = L
        # import pdb; pdb.set_trace()
        if mixture is not None:
            sigma_min = min(np.min(np.linalg.svd(X, compute_uv=False)) for i in range(n) for X in [mixture.Ms[:,i,:], mixture.Ms[:,:,i]])
            stats["sigma_min"] = sigma_min
    Ps_ = np.moveaxis(us, 1, 2)[:,:L]
    Qs_ = (ss[:,:L].T * vhs[:,:L].T).T

    A = np.zeros((2 * n * L, n ** 2))
    for j in range(n):
        A[L * j:L * (j + 1), n * j:n * (j + 1)] = Ps_[j]
        A[L * (n + j):L * (n + j + 1), j + n * (np.arange(n))] = -Qs_[j]

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    s_inc = s[::-1]
    s_inc_c = s_inc[:n-1]
    ratios = s_inc_c[1:] / s_inc_c[:-1]
    r = max(L, 1 + np.argmax(ratios * (s_inc_c[1:] < 0.1)))
    # import pdb; pdb.set_trace()
    if returnsval: return s_inc
    if verbose:
        print(s_inc[:4])
        with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
            print("singular values of A:", " ".join( f"{x:.5f}" for x in s_inc))
            print("ratios:                      ", " ".join( f"{x:.5f}" for x in ratios))
            print(f"suggest r={r} (vs L={L})")
    B = vh[-r:] # np.random.rand(r,r) @
    Bre = np.moveaxis(B.reshape((r, L, 2 * n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    Prods = Zs_ @ np.transpose(Ys_, axes=(0,2,1))
    ProdsInv = np.linalg.pinv(Prods)

    dists = []
    dists2 = []
    for i, j in itertools.combinations(range(n), 2):
        E = ProdsInv[i] @ Prods[j]
        Einv = ProdsInv[j] @ Prods[i]
        # randomized pseudoinverse test
        x = E @ np.random.rand(r, 1000)
        y = Einv @ E @ x
        dists.append(np.linalg.norm(x - y)**2 / (np.linalg.norm(x) * np.linalg.norm(y)))
        dists2.append(np.linalg.norm(x - y))

    lnk = linkage(np.array(dists), method="complete")
    groups = fcluster(lnk, sample_dist, criterion="distance") if sample_num is None else \
        fcluster(lnk, sample_num, criterion="maxclust")

    if verbose:
        for t in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]:
            groups_ = fcluster(lnk, t, criterion="distance")
            if verbose: print(groups_, t)

    dist_mtrx = squareform(dists)
    np.fill_diagonal(dist_mtrx, np.inf)
    parts = []

    if True:
        indMs = np.zeros((L, n, n))
        for g in range(max(groups)):
            states = np.where(groups == g+1)[0]
            group_dists = dist_mtrx[states][:,states]
            for i in states:
                j = np.argmin(dist_mtrx[i])
                R = reconstruct_at(i, j)
                p = PartialR(R, states, i, j)
                m = p.reconstruct()
                indMs[:,i] = m.Ms[:,i]
        if verbose and mixture is not None:
            print(states, "sample_ind:", Mixture.perm_dist(indMs[:, states], mixture.Ms[:, states]) / n)

    """
    if True: # if sample_all:
        Ms = np.zeros((L, n, n))
        for g in range(max(groups)):
            states = np.where(groups == g+1)[0]
            colMs, colS = [], []
            for i, j in itertools.combinations(states, 2):
                R = reconstruct_at(i, j)
                p = PartialR(R, states, i, j)
                m = p.reconstruct()
                colMs.append(m.Ms[:,states])
                colS.append(m.S[:,states])
            comMs = np.median(colMs, axis=0)
            comMs2 = find_representative(colMs)
            comS = np.median(colMs, axis=0)
            if verbose:
                print(states, "sample_all (median):   ", Mixture.perm_dist(comMs[:, states], mixture.Ms[:, states]) / n)
                print(states, "sample_all (find_repr):", Mixture.perm_dist(comMs2[:, states], mixture.Ms[:, states]) / n)
    """

    if True: # else:
        for g in range(max(groups)):
            states = np.where(groups == g+1)[0]
            group_dists = dist_mtrx[states][:,states]
            if len(states) > 1:
                a, b = np.unravel_index(np.argmin(group_dists), group_dists.shape)
                i, j = states[a], states[b]
                if pair_selection == "best":
                    best_pair = None
                    best_dist = np.inf
                    partial_sample = sample.restrict_to(states)
                    for i, j in itertools.combinations(states, 2):
                        R = reconstruct_at(i, j)
                        p = PartialR(R, states, i, j)
                        m = p.reconstruct().restrict_to(states)
                        m = Mixture(np.abs(m.S), np.abs(m.Ms))
                        m.normalize()
                        partial_distribution = Distribution.from_mixture(m, 3)
                        dist = partial_sample.dist(partial_distribution)
                        # print(i, j, dist)
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (i, j)
                    i, j = best_pair
                elif pair_selection == "rnd":
                    i, j = np.random.choice(states, 2, replace=False)
                elif pair_selection == "worst":
                    a, b = np.unravel_index(np.argmax(group_dists), group_dists.shape)
                    i, j = states[a], states[b]
            else:
                [i] = states
                j = np.argmin(dist_mtrx[i])
            if verbose: print(f"=== group {g}: {states} i={i} j={j} {'='*50}")
            R = reconstruct_at(i, j)
            parts.append(PartialR(R, states, i, j))

        if len(parts) > 1:
            S, Ms = combine(parts)
        else:
            m = parts[0].reconstruct()
            S, Ms = m.S, m.Ms

    if mixture is not None and verbose:
        for p in parts:
            p_mixture = p.reconstruct()
            print(p.states, "combined Ms:", Mixture.perm_dist(Ms[:, list(p.states)], mixture.Ms[:, list(p.states)]) / n,
                  "[vs] part Ms:", Mixture.perm_dist(p_mixture.Ms[:, list(p.states)], mixture.Ms[:, list(p.states)]) / n)
            # print(p.states, f"combined Ms:", np.linalg.norm((Ms - mixture.Ms)[:, list(p.states)].flatten(), ord=1) / f,
            #       "[vs] part Ms:", np.linalg.norm((p_mixture.Ms - mixture.Ms)[:, list(p.states)].flatten(), ord=1) / f)
        """
        print("normalization:")
        m1 = Mixture(S.copy(), Ms.copy())
        m1_ = Mixture(S.copy(), Ms.copy())
        m1_.normalize()
        m2 = Mixture(np.abs(S), np.abs(Ms))
        m2_ = Mixture(np.abs(S), np.abs(Ms))
        m2_.normalize()
        m3 = Mixture(np.abs(S), np.abs(Ms))
        m3_ = Mixture(np.clip(S, 0, 1), np.clip(Ms, 0, 1))
        m3_.normalize()
        m4 = Mixture(np.abs(S), np.abs(Ms))
        m4.normalize()
        m4_ = em_refine(m4, p.states)
        print(np.round(np.sum(m1.Ms, axis=2), 4))
        print(np.round(np.sum(m2.Ms, axis=2), 4))
        print(np.round(np.sum(m3.Ms, axis=2), 4))
        print("      () recov-err:", Mixture.recovery_error(m1, mixture))
        print("     (n) recov-err:", Mixture.recovery_error(m1_, mixture))
        print("   (abs) reocv-err:", Mixture.recovery_error(m2, mixture))
        print(" (abs,n) reocv-err:", Mixture.recovery_error(m2_, mixture))
        print("(abs,em) recov-err:", Mixture.recovery_error(m4_, mixture))
        # print("     (0) reocv-err:", Mixture.perm_dist(m3.Ms, mixture.Ms) / n)
        # print("   (0,n) reocv-err:", Mixture.perm_dist(m3_.Ms, mixture.Ms) / n)
        print("   (abs) tv_dist:  ", distribution.dist(Distribution.from_mixture(m2, 3)))
        print(" (abs,n) tv_dist:  ", distribution.dist(Distribution.from_mixture(m2_, 3)))
        # print("     (0) tv_dist:  ", distribution.dist(Distribution.from_mixture(m3, 3)))
        # print("   (0,n) tv_dist:  ", distribution.dist(Distribution.from_mixture(m3_, 3)))
        print("(abs,em) tv_dist:  ", distribution.dist(Distribution.from_mixture(m4_, 3)))
        """

    S, Ms = np.abs(S), np.abs(Ms)
    learned_mixture = Mixture(S, Ms)
    if em_refine_max_iter > 0:
        learned_mixture = em_refine(learned_mixture, em_refine_max_iter=em_refine_max_iter)
    learned_mixture.normalize()

    return learned_mixture













def svd_learn_new_(sample, n, L, compress=True, verbose=None, mixture=None, sample_dist=1, sample_num=None):

    class PartialR:
        def __init__(self, R, states, i, j):
            self.R = R
            self.Rinv = np.linalg.pinv(R)
            self.states = set(states)
            self.i = i
            self.j = j

        def reconstruct(self):
            Ys = self.R @ Ys_
            Ps = Ys @ Ps_
            S = np.real(np.diagonal(self.R @ Zs_ @ np.transpose(Ys_, axes=(0,2,1)) @ self.R.T, axis1=1, axis2=2).T)
            Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
            return Mixture(S, Ms)

    def reconstruct_at(i, j):
        E = ProdsInv[i] @ Prods[j]
        eigs, w = np.linalg.eig(E)
        # mask = eigs > 1e-5
        mask = np.argpartition(eigs, -L)[-L:]
        R_ = w[:, mask]
        d, _, _, _ = np.linalg.lstsq((R_.T @ Ys_[i] @ Ps_[i]).T, Os[i] @ np.ones(n), rcond=None)
        R = np.diag(d) @ R_.T
        return R
        """
        Ys = R @ Ys_
        Ps = Ys @ Ps_
        S = np.real(np.diagonal(R @ Zs_ @ np.transpose(Ys_, axes=(0,2,1)) @ R.T, axis1=1, axis2=2).T)
        Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
        if np.any(np.all(np.abs(np.linalg.pinv(R).T) < 1e-3, axis=1)):
            assert (False)
        Rinv = np.linalg.pinv(R).T
        if np.any([ np.allclose(x, y) for x, y in itertools.combinations(Rinv, 2) ]):
            assert (False)
        return S, Ms, np.linalg.pinv(R)
        """

    def combine(parts, compress=True):
        collectedRinv = np.real(np.vstack(list(p.Rinv.T for p in parts)))

        # dists = pdist(collectedRinv / np.linalg.norm(collectedRinv, axis=0))
        dists = np.zeros(len(collectedRinv) * (len(collectedRinv) - 1) // 2)
        for k, (i1, i2) in enumerate(itertools.combinations(range(len(collectedRinv)), 2)):
            dists[k] = np.linalg.norm(collectedRinv[i1] - collectedRinv[i2])**2 / \
                       (np.linalg.norm(collectedRinv[i1]) * np.linalg.norm(collectedRinv[i2]))
            p1, p2 = i1 // L, i2 // L
            if p1 == p2:
                dists[k] = 10
            elif parts[p1].i == parts[p2].j or parts[p1].j == parts[p2].i:
                dists[k] /= 10

        # fcluster seems buggy, so here's a quick fix
        dist_mtrx = squareform(dists + 1e-10 * np.random.rand(*dists.shape))
        double_dists = [0 if i//2 == j//2 else dist_mtrx[i//2, j//2]
                        for i, j in itertools.combinations(range(2 * len(dist_mtrx)), 2)]
        lnk = linkage(double_dists, method="complete")
        double_groups = fcluster(lnk, r, criterion="maxclust") - 1
        groups = np.array([g for i, g in enumerate(double_groups) if i%2])
        assert(max(groups)+1 == r)

        if verbose:
            from scipy.cluster.hierarchy import dendrogram
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            plt.figure(1, figsize=(15, 10))
            plt.subplot(211)
            dendrogram(lnk)
            tsne = TSNE(n_components=2, perplexity=4)
            tsne_results = tsne.fit_transform(collectedRinv)
            plt.subplot(212)
            for l in range(r):
                plt.scatter(tsne_results[groups==l, 0], tsne_results[groups==l, 1], s=100)
            plt.show()

        combinedRinv = np.zeros((r, r))
        for l in range(r):
            cluster = collectedRinv[groups==l]
            # intra_dists = np.linalg.norm(np.linalg.norm(cluster - cluster[:,None], axis=1), axis=1)
            # intra_dists = [np.linalg.norm(cluster - x) for x in cluster]
            intra_dists = np.sum(squareform(dists)[groups==l][:,groups==l], axis=0)
            center = cluster[np.argmin(intra_dists)]
            combinedRinv[l] = center
            if verbose:
                with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
                    avg_dist = np.average(
                        list(np.linalg.norm(x - y, ord=1) for x, y in itertools.combinations(cluster, 2))) if len(
                        cluster) > 1 else 0
                    cen_dist = np.linalg.norm(cluster - center, ord=1, axis=1)
                    print("-" * 10,
                          f"label={l} (size={len(cluster)}, d={avg_dist:.5f}) dist from center: avg={np.average(cen_dist):.5f} max={np.max(cen_dist):.5f}",
                          "-" * 10)
                    print("\n".join([
                        f"{'>' if np.allclose(collectedRinv[i], center) else ' '} {parts[i//L].i:2d} {parts[i//L].j:2d} ({i % L}) {x}"
                        for i, x in
                        zip(np.where(groups==l)[0], str(cluster).split("\n"))]))  # where(labels==l)

        """
        for _ in range(5):
            try:
                # centroids, labels = kmeans2(collectedRinv, r, minit="++", missing="raise")
                if verbose:
                    from scipy.cluster.hierarchy import dendrogram
                    import matplotlib.pyplot as plt
                    plt.figure(1, figsize=(15, 10))
                    plt.subplot(211)
                    dendrogram(lnk)

                    from sklearn.manifold import TSNE
                    import sys
                    tsne = TSNE(n_components=2, perplexity=4, metric="cityblock")
                    tsne_results = tsne.fit_transform(collectedRinv)
                    plt.subplot(212)

                    print(f"combinedRinv.shape={combinedRinv.shape}", "~"*50)
                    print(*[list(p.states) for p in parts])
                    with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
                        print("\n".join([f"{groups[i]} {x}"
                                     for i, x in zip(range(len(groups)), str(collectedRinv).split("\n"))]))

                        for l in range(r):
                            cluster = collectedRinv[groups==l+1]
                            center = cluster[
                                np.argmin(list(np.linalg.norm((cluster - x).flatten(), ord=1) for x in cluster))]
                            centroids[l] = center # np.mean(cluster, axis=0)
                            plt.scatter(tsne_results[labels == l, 0], tsne_results[labels == l, 1], s=100)

                        plt.subplot(313)
                        for l in range(r):
                            plt.scatter(tsne_results[groups==l+1, 0], tsne_results[groups==l+1, 1], s=100)
                            cluster = collectedRinv[groups == l + 1]  # collectedRinv[labels==l]
                            avg_dist = np.average(
                                list(np.linalg.norm(x - y, ord=1) for x, y in itertools.combinations(cluster, 2))) if len(
                                cluster) > 1 else 0
                            cen_dist = np.linalg.norm(cluster - centroids[l], ord=1, axis=1)
                            print("-" * 10,
                                  f"label={l} (size={len(cluster)}, d={avg_dist:.5f}) C: avg={np.average(cen_dist):.5f} max={np.max(cen_dist):.5f}",
                                  "-" * 10)
                            print(centroids[l])
                            print("\n".join([f"{'>' if np.allclose(collectedRinv[i], centroids[l]) else ' '} ({parts[i // L][4]:2d}&{parts[i // L][5]:2d}: {i % L}) {x}"
                                             for i, x in zip(np.where(groups == l + 1)[0], str(cluster).split("\n"))]))  # where(labels==l)
                            # print(cluster)
                            center = cluster[
                                np.argmin(list(np.linalg.norm((cluster - x).flatten(), ord=1) for x in cluster))]
                            # centroids[l] = center

                    plt.show()
                    # sys.exit()
                break
            except ClusterError:
                pass
        else:
            warnings.warn("Could not find clustering")
            centroids = collectedRinv[:r]
            centroids = np.vstack((centroids, np.tile(collectedRinv[-1], (r - len(centroids), 1))))
        combinedRinv = centroids
        """
        assert(len(combinedRinv) == r)
        R = np.linalg.pinv(combinedRinv.T)

        def asgn_mtrx2(mass):
            A = cp.Variable((L, r), boolean=True)
            # objective = cp.Minimize(cp.max(A @ mass))
            # objective = cp.Minimize(cp.sum_squares(A @ mass))
            objective = cp.Minimize(cp.sum(cp.max(A @ mass, axis=0)))
            constraint = cp.sum(A, axis=0) == 1
            prob = cp.Problem(objective, [constraint])
            try:
                # cvxopt.glpk.options["msg_lev"] = "GLP_MSG_ON"
                solve_time = time.time()
                print("<", end="", flush=True)
                # prob.solve(solver=cvxopt.glpk)
                # prob.solve(solver="GLPK_MI", tmlim=0.05, verbose=False) # GLPK_MI_params={"tm_lim": 0.001}
                prob.solve(verbose=False, solver="CBC", maximumSeconds=5)
                # prob.solve(verbose=False, solver="GLPK", maximumSeconds=1)
                # options = ["--tmlim"
                print(">", end="", flush=True)
                # assert(time.time() - solve_time < 1)
                assert(A.value.shape == (L, r))
                return A.value
                # if problem.status == 'optimal': .... else: ....
            except Exception as e:
                print("\n\n\t\t-- -- EXCEPTION -- --\n\n")
                print(e)
                print("\n\n\t\t-- --           -- --\n\n")
                return np.tile(np.eye(L), r // L + 1)[:,:r]

        def asgn_mtrx(mass):
            asgn = np.zeros(r ,dtype=int)
            asgn_mass = np.zeros((L, n))
            asgn_cost = 0
            min_asgn_cost = np.inf
            min_asgn = np.zeros_like(asgn)

            def assign(i, j):
                nonlocal asgn_cost
                asgn[i] = j
                cost = np.linalg.norm(np.minimum(asgn_mass[j], mass[i])) ** 2
                asgn_cost += cost
                asgn_mass[j] += mass[i]
                return (i, j, cost)

            def unassign(i, j, cost):
                nonlocal asgn_cost
                asgn_cost -= cost
                asgn_mass[j] -= mass[i]

            # greedy initialization
            xs = []
            for i in range(len(mass)):
                j = np.argmin(np.linalg.norm(asgn_mass, axis=1))
                xs.append(assign(i, j))
            min_asgn[:] = asgn
            min_asgn_cost = asgn_cost
            for x in xs:
                unassign(*x)

            solve_time = time.time()
            def comp_rec(i):
                nonlocal asgn_cost, min_asgn_cost
                if asgn_cost > min_asgn_cost:
                    return False
                if i >= len(mass):
                    min_asgn[:] = asgn
                    min_asgn_cost = asgn_cost
                    return time.time() > solve_time + 1
                js = list(range(L))
                np.random.shuffle(js)
                for j in js:
                    """
                    asgn[i] = j
                    cost = np.linalg.norm(np.minimum(asgn_mass[j], mass[i]))**2
                    asgn_cost += cost
                    asgn_mass[j] += mass[i]
                    comp_rec(i+1)
                    asgn_cost -= cost
                    asgn_mass[j] -= mass[i]
                    """
                    x = assign(i, j)
                    if comp_rec(i + 1): return True
                    unassign(*x)

            if comp_rec(0) and verbose: print("timeout for asgnm")
            mtrx = np.zeros((L, r), dtype=int)
            mtrx[min_asgn, range(r)] = 1
            return mtrx

        def compression_mtrx(assoc):
            def comp_rec(tmp, assoc, i, ass):
                if i < len(assoc):
                    for j in range(L):
                        if np.all(tmp[j] + assoc[i] <= 1):
                            tmp[j] += assoc[i]
                            compressable = comp_rec(tmp, assoc, i+1, ass)
                            tmp[j] -= assoc[i]
                            if compressable:
                                ass[i] = j
                                return True
                    return False
                else:
                    return True
            ass = np.zeros(r, dtype=int)
            tmp = np.zeros((L, n))
            assert(comp_rec(tmp, assoc, 0, ass))
            comp = np.zeros((L, r), dtype=int)
            comp[ass, range(r)] = 1
            return comp

        Ys = R @ Ys_
        """
        cc_assoc = (np.linalg.norm(Ys, axis=2) > 1e-2).astype(int).T
        if verbose: print(cc_assoc)
        comp = compression_mtrx(cc_assoc) if compress else np.eye(r)
        """
        comp = asgn_mtrx(np.linalg.norm(Ys, axis=2, ord=1).T)
        if verbose: print(comp)
        compressedR = comp @ R
        if verbose:
            print(f"compressedR.shape = {compressedR.shape} (ideal is {L},{r})")

        Ys = compressedR @ Ys_
        Zs = compressedR @ Zs_
        Ps = Ys @ Ps_
        S = np.real(np.diagonal(Zs @ np.transpose(Ys, axes=(0,2,1)), axis1=1, axis2=2).T)
        Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
        return S, Ms

    Os = np.moveaxis(sample.all_trail_probs(), 1, 0)
    us, ss, vhs = np.linalg.svd(Os)
    Ps_ = np.moveaxis(us, 1, 2)[:,:L]
    Qs_ = (ss[:,:L].T * vhs[:,:L].T).T

    A = np.zeros((2 * n * L, n ** 2))
    for j in range(n):
        A[L * j:L * (j + 1), n * j:n * (j + 1)] = Ps_[j]
        A[L * (n + j):L * (n + j + 1), j + n * (np.arange(n))] = -Qs_[j]

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    # thresh = 0.0001
    # r = sum(s < thresh)
    s_inc = s[::-1]
    ratios = s_inc[1:] / s_inc[:-1]
    r = max(L, 1 + np.argmax(ratios * (s_inc[1:] < 0.1)))
    if verbose:
        with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
            print("singular values of A:", " ".join( f"{x:.5f}" for x in s_inc))
            print("ratios:                      ", " ".join( f"{x:.5f}" for x in ratios))
            print(f"suggest r={r} (vs L={L})")
    B = np.random.rand(r,r) @ vh[-r:]
    Bre = np.moveaxis(B.reshape((r, L, 2 * n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    Prods = Zs_ @ np.transpose(Ys_, axes=(0,2,1))
    ProdsInv = np.linalg.pinv(Prods)

    dists = []
    dists2 = []
    for i, j in itertools.combinations(range(n), 2):
        # for i, j in [(a, b), (b, a)]:
        E = ProdsInv[i] @ Prods[j]
        Einv = ProdsInv[j] @ Prods[i]
        # randomized pseudoinverse test
        x = E @ np.random.rand(r, 50)
        y = Einv @ E @ x
        dists.append(np.linalg.norm(x - y)**2 / (np.linalg.norm(x) * np.linalg.norm(y)))
        dists2.append(np.linalg.norm(x - y))
        # dists.append(np.mean(np.linalg.norm(Einv @ E @ x - x, axis=0)))

    lnk = linkage(np.array(dists), method="complete")
    groups = fcluster(lnk, sample_dist, criterion="distance") if sample_num is None else \
             fcluster(lnk, sample_num, criterion="maxclust")

    if verbose:
        for t in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]:
            groups_ = fcluster(lnk, t, criterion="distance")
            if verbose: print(t, groups_)

    dist_mtrx = squareform(dists)
    np.fill_diagonal(dist_mtrx, np.inf)
    parts = []

    if sample_num is not None and max(groups) < sample_num:
        for i in range(n):
            d, q = divmod(sample_num, max(groups))
            k = d+1 if i < q else d
            js = np.argpartition(dist_mtrx[i], k)[:k]
            for j in js:
                R = reconstruct_at(i, j)
                parts.append(PartialR(R, [i], i, j))

    else:
        for g in range(max(groups)):
            states = np.where(groups == g+1)[0]
            group_dists = dist_mtrx[states][:,states]
            if len(states) > 1:
                a, b = np.unravel_index(np.argmin(group_dists), group_dists.shape)
                i, j = states[a], states[b]
            else:
                [i] = states
                j = np.argmin(dist_mtrx[i])
                # i, j = min(i, j), max(i, j)
            if verbose: print(f"=== group {g}: {states} i={i} j={j} {'='*50}")
            # i = np.random.choice(states)
            # d_mtrx[i][i] = np.inf
            # j = np.argmin(d_mtrx[i])
            R = reconstruct_at(i, j)
            parts.append(PartialR(R, states, i, j))
        """
        if mixture is not None:
            part_mixture = Mixture(S, Ms)
            if verbose:
                print(Ms.shape)
                rows, cols = linear_sum_assignment(list(list(
                    np.linalg.norm(((Ms[i] - mixture.Ms[j])[states][:,states]).flatten(), ord=1)
                      for i in range(L)) for j in range(L)))
                with np.printoptions(precision=3, suppress=True, linewidth=np.inf):
                    print(np.sum(np.abs(part_mixture.Ms[rows] - mixture.Ms[cols]), axis=0))
                print(part_mixture)
        """
        # parts.append((states, S, Ms, R, i, j))

    S, Ms = combine(parts)
    if mixture is not None and verbose:
        for p in parts:
            p_mixture = p.reconstruct()
            print(p.states, f"combined Ms:", np.linalg.norm((Ms - mixture.Ms)[:, list(p.states)].flatten(), ord=1),
                  #"S:", np.linalg.norm((S - mixture.S)[:, states].flatten(), ord=1),
                  "[vs] part Ms:", np.linalg.norm((p_mixture.Ms - mixture.Ms)[:, list(p.states)].flatten(), ord=1),
                  #"S:", np.linalg.norm((p[1] - mixture.S)[:, states].flatten(), ord=1)
                  )
    S, Ms = np.abs(S), np.abs(Ms)
    learned_mixture = Mixture(S, Ms)
    learned_mixture.normalize()
    return learned_mixture















def svd_learn6(sample, n, L=None, verbose=None, mixture=None):
    start = time.time()

    def reconstruct_at(i, j):
        E = ProdsInv[i] @ Prods[j]
        eigs, w = np.linalg.eig(E)
        mask = eigs > 1e-5
        R_ = w[:, mask]
        d, _, _, _ = np.linalg.lstsq((R_.T @ Ys_[i] @ Ps_[i]).T, Os[i] @ np.ones(n), rcond=None)
        R = np.diag(d) @ R_.T
        print(f"==== at {i},{j} ====")
        print(np.linalg.pinv(R))
        Ys = R @ Ys_
        Ps = Ys @ Ps_
        S = np.real(np.diagonal(R @ Zs_ @ np.transpose(Ys_, axes=(0,2,1)) @ R.T, axis1=1, axis2=2).T)
        Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
        return S, Ms, np.linalg.pinv(R)

    """
    def combine4(parts):
        n_parts = len(parts)
        com_assoc = np.zeros((r, n_parts), dtype=int)
        row_assoc = np.zeros((n_parts, L), dtype=int)
        combinedRinv = np.empty((0, r))
        for p, (_, _, _, Rinv) in enumerate(parts):
            for i, row in enumerate(Rinv.T):
                d = np.linalg.norm(combinedRinv - row, axis=1)
                if not np.any(d < 1e-5):
                    ix = len(combinedRinv)
                    combinedRinv = np.vstack((combinedRinv, row))
                else:
                    ix = np.argmin(d)
                com_assoc[ix, p] = 1
                row_assoc[p, i] = ix
        print(com_assoc)

        def compress(assoc, i, ass):
            if i < len(ass):
                for j in range(L):
                    if np.all(assoc[j] + assoc[i] <= 1):
                        assoc[j] += assoc[i]
                        compressable = compress(assoc, i+1, ass)
                        assoc[j] -= assoc[i]
                        if compressable:
                            ass[i] = j
                            return True
                return False
            else:
                return True

        ass = np.arange(r)
        assert(compress(com_assoc, L, ass))
        print(ass)
        compressedRinv = np.zeros((L, r))
        for p, (_, _, _, Rinv) in enumerate(parts):
            for i, row in enumerate(Rinv.T):
                compressedRinv[ass[row_assoc[p, i]]] += row

        R = np.linalg.pinv(compressedRinv.T)
        print(compressedRinv)
        Ys = R @ Ys_
        Ps = Ys @ Ps_
        S = np.real(np.diagonal(R @ Zs_ @ np.transpose(Ys_, axes=(0,2,1)) @ R.T, axis1=1, axis2=2).T)
        Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
        return S, Ms
    """

    def combine(parts):
        combinedRinv = np.empty((0, r))
        for (_, _, _, Rinv) in parts:
            for col in Rinv.T:
                d = np.linalg.norm(combinedRinv - col, axis=1)
                if not np.any(d < 1e-5):
                    combinedRinv = np.vstack((combinedRinv, col))
        R = np.linalg.pinv(combinedRinv.T)

        def compression_mtrx(assoc):
            def comp_rec(assoc, i, ass):
                if i < len(ass):
                    for j in range(L):
                        if np.all(assoc[j] + assoc[i] <= 1):
                            assoc[j] += assoc[i]
                            compressable = comp_rec(assoc, i+1, ass)
                            assoc[j] -= assoc[i]
                            if compressable:
                                ass[i] = j
                                return True
                    return False
                else:
                    return True

            ass = np.arange(r)
            assert(comp_rec(assoc, L, ass))
            comp = np.zeros((L, r))
            for i, j in enumerate(ass):
                comp[j, i] = 1
            return comp

        Ys = R @ Ys_
        cc_assoc = (np.linalg.norm(Ys, axis=2) > 1e-5).astype(int).T
        comp = compression_mtrx(cc_assoc) if compress else np.eye(r)
        compressedR = comp @ R

        Ys = compressedR @ Ys_
        Zs = compressedR @ Zs_
        Ps = Ys @ Ps_
        S = np.real(np.diagonal(Zs @ np.transpose(Ys, axes=(0,2,1)), axis1=1, axis2=2).T)
        Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
        return S, Ms

    def combine3(parts):
        combinedRinv = np.empty((0, r))
        for (states, _, _, Rinv) in parts:
            with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
                print(states)
                print(np.real(Rinv.T))
            for col in Rinv.T:
                d = np.linalg.norm(combinedRinv - col, axis=1)
                if not np.any(d < 1e-5):
                    combinedRinv = np.vstack((combinedRinv, col))
        print(combinedRinv.T)
        R = np.linalg.pinv(combinedRinv.T)
        Ys = R @ Ys_
        Ps = Ys @ Ps_
        S = np.real(np.diagonal(R @ Zs_ @ np.transpose(Ys_, axes=(0,2,1)) @ R.T, axis1=1, axis2=2).T)
        Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
        return S, Ms

    """
    def combine2(parts):
        for (states1, S1, Ms1), (states2, S2, Ms2) in itertools.combinations(parts, 2):
            dists = np.zeros((L,L))
            for l1, l2 in itertools.product(range(L), repeat=2):
                x1 = np.sum(np.abs(Ms1[l1][states1][:,states2] - Ms2[l2][states1][:,states2]))
                x2 = np.sum(np.abs(Ms1[l1][states2][:,states1] - Ms2[l2][states2][:,states1]))
                d = x1 + x2
                dists[l1,l2] = d
            print(f"==== {states1} ~ {states2} ====")
            with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
                print(dists)

    def combine(parts):
        combinedS = np.empty((L, n))
        combinedS[:] = np.NaN
        combinedMs = np.empty((L, n, n))
        combinedMs[:] = np.NaN
        while parts:
            dists = []
            perms = []
            for states, S, Ms in parts:
                ds = np.zeros((L, L))
                for l1, l2 in itertools.product(range(L), repeat=2):
                    x = np.abs((Ms[l1] - combinedMs[l2])[:,states])
                    # d = x1 + x2
                    x = np.abs(Ms[l1] - combinedMs[l2])
                    d = x[~np.isnan(x)]
                    ds[l2, l1] = sum(d)
                    # ds[l2, l1] = sum(np.log(1 + d))
                rows, cols = linear_sum_assignment(ds)
                dists.append(sum(ds[rows,cols]))
                perms.append(cols)
            ix = np.argmin(dists)
            states, S, Ms = parts[ix]
            perm = perms[ix]
            del parts[ix]
            combinedS[:,states] = S[perm][:,states]
            combinedMs[:,states] = Ms[perm][:,states]
        return combinedS, combinedMs
    """

    Os = np.moveaxis(sample.all_trail_probs(), 1, 0)
    us, ss, vhs = np.linalg.svd(Os)
    Ps_ = np.moveaxis(us, 1, 2)[:,:L]
    Qs_ = (ss[:,:L].T * vhs[:,:L].T).T

    A = np.zeros((2 * n * L, n ** 2))
    for j in range(n):
        A[L * j:L * (j + 1), n * j:n * (j + 1)] = Ps_[j]
        A[L * (n + j):L * (n + j + 1), j + n * (np.arange(n))] = -Qs_[j]

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    print("singular values of A:", *[f"{x:.5f}" for x in s[::-1]])
    thresh = 1e-5
    r = sum(s < thresh)
    print(f"{r} are less than {thresh} (ideal is {L})")
    B = np.random.rand(r,r) @ vh[-r:]
    Bre = np.moveaxis(B.reshape((r, L, 2 * n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    """
    # for debugging
    svds = [np.linalg.svd(Os[j], full_matrices=True) for j in range(n)]
    realZs = np.zeros((n, L, L))
    realYs = np.zeros((n, L, L))
    for j, (u, s, vh) in enumerate(svds):
        P_ = Ps_[j]
        P = mixture.S * mixture.Ms[:, :, j]
        X = np.linalg.lstsq(P.T, P_.T, rcond=None)[0].T
        Y = np.linalg.inv(X)
        Z = np.diag(mixture.S[:, j]) @ X.T
        realYs[j] = Y
        realZs[j] = Z

    originB = np.array([
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1]
    ])

    # originB = np.hstack(np.eye(L) for _ in range(n))
    # originB = np.vstack((originB, [1,0]*2 + [0,0]*2))

    # originB = np.vstack((originB,
    #                      [1,0]*4 + [0,0]*4,
    #                      [1,0]*2 + [0,0]*4 + [1,0]*2))

    originB = np.hstack((originB, originB))
    D = scipy.linalg.block_diag(*realYs, *realZs)
    realB = originB @ D
    realR = np.linalg.lstsq(B.T, realB.T, rcond=None)[0].T
    print(realR)
    """

    Prods = Zs_ @ np.transpose(Ys_, axes=(0,2,1))
    ProdsInv = np.linalg.pinv(Prods)

    dists = []
    for i, j in itertools.combinations(range(n), 2):
        E = ProdsInv[i] @ Prods[j]
        Einv = ProdsInv[j] @ Prods[i]

        # randomized pseudoinverse test
        x = E @ np.random.rand(r)
        d = np.linalg.norm(Einv @ E @ x - x)
        dists.append(d)
        print(f"{i}~{j}={d}")

    lnk = linkage(dists, method="complete")
    groups = fcluster(lnk, 1e-5, criterion="distance")

    d_mtrx = squareform(dists)
    parts = []
    for g in range(max(groups)):
        states = np.where(groups == g+1)[0]
        print(f"=== group {g}: {states} {'='*50}")
        i = np.random.choice(states)
        d_mtrx[i][i] = np.inf
        j = np.argmin(d_mtrx[i])
        S, Ms, R = reconstruct_at(i, j)
        parts.append((states, S, Ms, R))

    # combine4(parts)
    S, Ms = combine(parts, compress=True)
    # combinedS, combinedMs = combine3(parts)
    S, Ms = np.abs(S), np.abs(Ms)
    learned_mixture = Mixture(S, Ms)
    learned_mixture.normalize()
    learned_mixture.print()

    print(f"recovered mixture in {time.time() - start}s")
    return learned_mixture


def svd_learn5(sample, n, L=None, sv_threshold=0.05, verbose=None, mixture=None):
    Os = np.moveaxis(sample.all_trail_probs(), 1, 0)
    svds = [np.linalg.svd(Os[j], full_matrices=True) for j in range(n)]

    Ps_ = np.zeros((n, L, n))
    Qs_ = np.zeros((n, L, n))
    realPs = np.zeros((n, L, n))
    realQs = np.zeros((n, L, n))
    realXs = np.zeros((n, L, L))
    realZs = np.zeros((n, L, L))
    realYs = np.zeros((n, L, L))
    for j, (u, s, vh) in enumerate(svds):
        Ps_[j, 0:min(n, L), :] = u[:, 0:L].T
        Qs_[j, 0:min(n, L), :] = (np.diag(s) @ (vh))[0:L, :]
        P_ = Ps_[j]
        Q_ = Qs_[j]
        P = mixture.S * mixture.Ms[:, :, j]
        Q = mixture.S[:, j][:, np.newaxis] * mixture.Ms[:, j, :]
        X = np.linalg.lstsq(P.T, P_.T, rcond=None)[0].T
        Y = np.linalg.inv(X)
        Z = np.diag(mixture.S[:, j]) @ X.T
        realPs[j] = P
        realQs[j] = Q
        realXs[j] = X
        realYs[j] = Y
        realZs[j] = Z

    A = np.zeros((2 * n * L, n ** 2))
    realA = np.zeros((2 * n * L, n ** 2))
    for j in range(n):
        A[L * j:L * (j + 1), n * j:n * (j + 1)] = Ps_[j]
        A[L * (n + j):L * (n + j + 1), j + n * (np.arange(n))] = -Qs_[j]
        realA[L * j:L * (j + 1), n * j:n * (j + 1)] = realPs[j]
        realA[L * (n + j):L * (n + j + 1), j + n * (np.arange(n))] = -realQs[j]

    D = np.zeros((2 * n * L, 2 * n * L))
    for j in range(n):
        D[L * j:L * (j + 1), L * j:L * (j + 1)] = np.linalg.inv(realYs[j])
        D[L * (j + n):L * (j + n + 1), L * (j + n):L * (j + n + 1)] = np.linalg.inv(realZs[j])

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    print("singular values of A:", *[f"{x:.5f}" for x in s])
    r = sum(s < 1e-5)
    print(f"{r} are about 0 (ideal is {L})")
    B = np.random.rand(r,r) @ vh[-r:]
    print("shape of co-kernel is", B.shape)
    Bre = np.moveaxis(B.reshape((r, L, 2 * n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2 * n]

    originB = np.hstack(np.eye(L) for _ in range(n))
    originB = np.vstack((originB, [0,0]*2 + [1,0]*2))
    originB = np.hstack((originB, originB))
    D = scipy.linalg.block_diag(*realYs, *realZs)

    # realB = np.hstack((np.hstack(realYs), np.hstack(realZs)))
    # ccMask = [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4
    # realB = np.vstack((realB, ccMask * realB[0]))
    realB = originB @ D
    realR = np.linalg.lstsq(B.T, realB.T, rcond=None)[0].T

    Rinv = [ np.linalg.inv(realR) @ originB[:,L*i:L*(i+1)] for i in range(n) ]

    pinv = np.linalg.pinv
    print(pinv(np.linalg.inv(realR) @ originB[:, :2]) @ np.linalg.inv(realR))
    print("@")
    print(originB[:, 4:6])
    print("=")
    print(pinv(np.linalg.inv(realR) @ originB[:, :2]) @ np.linalg.inv(realR) @ originB[:, 4:6])

    Prods = [Zs_[i] @ Ys_[i].T for i in range(n)]
    ProdsInv = [np.linalg.pinv(P) for P in Prods]

    """
    ranks = np.zeros((n + 1, n + 1), dtype=int)
    ranks[0:n + 1, 0] = range(n + 1)
    ranks[0, 0:n + 1] = range(n + 1)
    pairs = []
    pairs2 = []
    for i, j in itertools.product(range(n), repeat=2):
        if i == j: continue
        s, _ = np.linalg.eig(ProdsInv[i] @ Prods[j])
        s = sorted(list(np.absolute(s)), reverse=True)
        rk = sum(np.absolute(s) > 1e-10)
        c = np.absolute(s[L]) / np.absolute(s[L - 1])
        ranks[i + 1, j + 1] = rk
        line = f"{i} {j}: {rk} ({np.absolute(s[rk - 1]) if rk > 0 else '-'} > {np.absolute(s[rk]) if rk < len(s) else '-'} : {c})"
        # print(line)
        with np.printoptions(suppress=True):
            print(i, j, np.round(s, 5), np.round(s[L - 1] / s[L - 2], 5))
        pairs.append((rk, np.absolute(s[rk - 1]), line))
        pairs2.append((c, line))
    print(" ----- SORTED --------------")
    pairs2.sort()
    for _, line in pairs2:
        print(line)
    print(" ----- MATRIX --------------")
    print(ranks)
    """

    combinedS = np.empty((L, n))
    combinedS[:] = np.NaN
    combinedMs = np.empty((L, n, n))
    combinedMs[:] = np.NaN
    for i, j in itertools.combinations(range(n), 2):
        print(f"= {i} {j} ", 50 * "=")
        E = ProdsInv[i] @ Prods[j]
        Einv = ProdsInv[j] @ Prods[i]

        tol = 1e-5
        eigs, w = np.linalg.eig(E)
        eigsInv = np.linalg.eigvals(Einv)
        eigsFromInv = np.divide(1, eigsInv, out=np.zeros_like(eigsInv), where=eigsInv > tol)

        d = np.linalg.norm(np.sort(eigs) - np.sort(eigsFromInv))
        print("(proximity)", d)
        if d > tol: continue

        mask = eigs > tol
        R_ = w[:, mask]
        d2, _, _, _ = np.linalg.lstsq((R_.T @ Ys_[i] @ Ps_[i]).T, Os[i] @ np.ones(n), rcond=None)
        R = np.diag(d2) @ R_.T

        Ys = R @ Ys_
        Zs = R @ Zs_
        Ps = Ys @ Ps_
        Qs = Zs @ Qs_
        S = np.diagonal(R @ Zs_ @ np.transpose(Ys_, axes=(0,2,1)) @ R.T, axis1=1, axis2=2).T
        MsP = np.transpose(Ps / S, axes=(1,2,0))
        MsQ = np.transpose(np.transpose(Qs, axes=(2,1,0)) / S, axes=(1,2,0))

        """
        with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
            print(np.allclose(MsP, MsQ))
            comps = []
            for perm in itertools.permutations(range(L)):
                comp = (np.abs(mixture.Ms - MsP[list(perm)]) < 1e-5).astype(int)
                comps.append((np.sum(comp), perm, comp))
            _, perm, comp = max(comps)
            print(np.clip(MsP[list(perm)], 0, 10))
            print(mixture.Ms)
            print(perm)
            print(comp)
        """

        states = [i, j]
        ds = np.zeros((L, L))
        for lnew, lold in itertools.product(range(L), repeat=2):
            print(f"---- {lnew} to {lold} ------------")
            x = np.abs(MsP[lnew] - combinedMs[lold])
            with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
                print(x)
            d = x[~np.isnan(x)]
            ds[lold, lnew] = sum(np.log(1 + d))
            print(ds[lold, lnew])
        print("-" * 100)
        _, perm = linear_sum_assignment(ds)
        print(ds)
        combinedMs[:,states] = MsP[perm][:,states]
        combinedS[:,states] = S[perm][:,states]

        with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
            print(perm)
            print(combinedS)
            print(combinedMs)

    combinedS = np.abs(combinedS)
    combinedMs = np.abs(combinedMs)
    learned_mixture = Mixture(combinedS, combinedMs)
    learned_mixture.normalize()
    learned_mixture.print()
    return learned_mixture

def find_connected_components(edge_list):
    es = {}
    for e in list(edge_list):
        for i, j in [e, e[::-1]]:
            if i not in es:
                es[i] = []
            if j not in es[i]:
                es[i].append(j)
    cc = []
    while es:
        for w in es.keys(): break
        component = [w]
        i = 0
        while i < len(component):
            v = component[i]
            for u in es[v]:
                if u not in component:
                    component.append(u)
            del es[v]
            i += 1
        cc.append(component)
    return cc


def svd_learn4(sample, n, L=None, sv_threshold=0.05, verbose=None, mixture=None):
    # assert(len(sample.trails) == n**sample.t_len)
    Os = np.moveaxis(sample.all_trail_probs(), 1, 0)

    svds = [ np.linalg.svd(Os[j], full_matrices=True) for j in range(n) ]

    if verbose:
        for i, (_, s, _) in enumerate(svds):
            print(f"{i}: {s[:L+1]} ...")

    if L is None:
        above_thresh = [ np.sum(s / s[0] > sv_threshold) for _, s, _ in svds ]
        L = int(np.median(above_thresh))
        if verbose is not None:
            for (_, s, _), t in zip(svds, above_thresh):
                print(s / s[0], t)
            print("Guessed L={}".format(L))

    Ps_ = np.zeros((n, L, n))
    Qs_ = np.zeros((n, L, n))
    Ps = np.zeros((n, L, n))
    Qs = np.zeros((n, L, n))
    Xs = np.zeros((n, L, L))
    Zs = np.zeros((n, L, L))
    Ys = np.zeros((n, L, L))
    for j, (u, s, vh) in enumerate(svds):
        Ps_[j, 0:min(n,L), :] = u[:, 0:L].T
        Qs_[j, 0:min(n,L), :] = (np.diag(s) @ (vh))[0:L, :]
        P_ = Ps_[j]
        Q_ = Qs_[j]
        P = mixture.S * mixture.Ms[:,:,j]
        Q = mixture.S[:,j][:,np.newaxis] * mixture.Ms[:,j,:]
        X = np.linalg.lstsq(P.T, P_.T)[0].T
        Y = np.linalg.inv(X)
        Z = np.diag(mixture.S[:,j]) @ X.T
        Ps[j] = P
        Qs[j] = Q
        Xs[j] = X
        Ys[j] = Y
        Zs[j] = Z

    A = np.zeros((2 * n * L, n**2))
    realA = np.zeros((2 * n * L, n**2))
    for j in range(n):
        A[L*j:L*(j+1), n*j:n*(j+1)] = Ps_[j]
        A[L*(n+j):L*(n+j+1), j+n*(np.arange(n))] = -Qs_[j]
        realA[L*j:L*(j+1), n*j:n*(j+1)] = Ps[j]
        realA[L*(n+j):L*(n+j+1), j+n*(np.arange(n))] = -Qs[j]

    D = np.zeros((2*n*L, 2*n*L))
    for j in range(n):
        D[L*j:L*(j+1), L*j:L*(j+1)] = np.linalg.inv(Ys[j])
        D[L*(j+n):L*(j+n+1), L*(j+n):L*(j+n+1)] = np.linalg.inv(Zs[j])

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    print("singular values of A:", *[f"{x:.5f}" for x in s ])
    k = sum(s < 1e-5)
    print(f"{k} are about 0 (ideal is {L})")
    B = vh[-k:]
    # B = np.random.rand(L, L) @ vh[-k:][:L] # <-
    # k = L # <-
    print("shape of co-kernel is", B.shape)
    Bre = np.moveaxis(B.reshape((k, L, 2*n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    realB = np.hstack((np.hstack(Ys), np.hstack(Zs)))
    # ccMask = [0] * 4 + [1] * 4 + [0] * 4 + [1] * 4
    # realB = np.vstack((realB, ccMask * realB[0]))
    realR = np.linalg.lstsq(B.T, realB.T)[0].T

    Prods = [ Zs_[i] @ Ys_[i].T for i in range(n) ]
    ProdsInv = [ np.linalg.pinv(P) for P in Prods ]

    # realR = None
    for i, P in enumerate(Prods):
        print(f"-{i}-----------------------")
        print(P)
        print(np.linalg.matrix_rank(P))
        s, _ = np.linalg.eig(P)
        print(s)
        print("-------------------------")
        # newR = np.linalg.lstsq(Ys_[i].T, Ys[i].T)[0].T
        # if realR is None or not np.allclose(realR, newR):
        #     print("newR:\n", newR)
        # realR = newR
        # assert(np.allclose(Ys[i], realR @ Ys_[i]))
        # assert(np.allclose(Zs[i], realR @ Zs_[i]))

    ranks = np.zeros((n+1, n+1), dtype=int)
    ranks[0:n+1,0] = range(n+1)
    ranks[0,0:n+1] = range(n+1)
    pairs = []
    pairs2 = []
    for i, j in itertools.product(range(n), repeat=2):
        if i == j: continue
        s, _ = np.linalg.eig(ProdsInv[i] @ Prods[j])
        s = sorted(list(np.absolute(s)), reverse=True)
        rk = sum(np.absolute(s) > 1e-10)
        c = np.absolute(s[L]) / np.absolute(s[L-1])
        ranks[i+1,j+1] = rk
        line = f"{i} {j}: {rk} ({np.absolute(s[rk-1]) if rk > 0 else '-'} > {np.absolute(s[rk]) if rk < len(s) else '-'} : {c})"
        # print(line)
        with np.printoptions(suppress=True):
            print(i, j, np.round(s, 5), np.round(s[L-1] / s[L-2], 5))
        pairs.append((rk, np.absolute(s[rk-1]), line))
        pairs2.append((c, line))
    print(" ----- SORTED --------------")
    pairs2.sort()
    for _, line in pairs2:
        print(line)
    print(" ----- MATRIX --------------")
    print(ranks)

    allS_ = []
    allMs_ = []
    for i, j in itertools.product(range(n), repeat=2): # itertools.combinations(range(n), 2):
        print(f"= {i} {j} ", 50 * "=")
        E = ProdsInv[i] @ Prods[j]
        s, R_ = np.linalg.eig(E)
        mask = s > 1e-5
        E_ = R_[:,mask]
        if i != j:
            # print(E)
            # compE = realR.T @ np.diag(1 / mixture.S[:,i]) @ np.diag(mixture.S[:,j]) @ np.linalg.pinv(realR).T
            # print(compE)
            # print("SAME" if np.allclose(E, compE) else "FAIL", flush=True)
            # import code; code.interact(local=locals())
            pass
        # print(R_.shape)
        # print(np.round(R_, 3))
        print(np.round(E_, 3))
        # print(R2_ @ np.diag(s[mask]) @ R2_.T)
        # print(R_ @ np.diag(s) @ np.linalg.inv(R_))
        print(np.linalg.norm(E_ @ np.diag(s[mask]) @ np.linalg.pinv(E_) - E))
        # print(E)
        print(s)
        print(np.absolute(s) > 1e-5)

        print(f"(r={k} > rank={sum(np.absolute(s) > 1e-10)} > L={L}) reconstructing...")

        print(R_)
        print(f"lstsq(({R_.T.shape} @ {Ys_[i].shape} @ {Ps_[i].shape}).T, {Os[i].shape} @ {np.ones(n).shape})")
        d, residuals, _, _ = np.linalg.lstsq((R_.T @ Ys_[i] @ Ps_[i]).T, Os[i] @ np.ones(n), rcond=None)
        print("d =", d, f"({residuals})")

        d2, _, _, _ = np.linalg.lstsq((E_.T @ Ys_[i] @ Ps_[i]).T, Os[i] @ np.ones(n), rcond=None)

        mask = np.abs(d) > 1e-5
        prevR = (np.diag(d) @ R_.T)[mask]
        R = np.diag(d2) @ E_.T
        print("R (prev) =")
        print(prevR)
        print("R (est) =")
        print(R)
        print(np.linalg.svd(R)[1])
        print("R (real) =")
        print(realR)

        n_rows = len(R)
        if n_rows < L:
            print(f"shape of R {R.shape} does not fit. Adding repeated rows")
            R = np.vstack([R, *([R[-1]] * (L - n_rows))])
        elif n_rows > L:
            print(f"too many rows in R {R.shape}. Ignore...")

        compYs = R @ Ys_

        if Ys.shape == compYs.shape:
            print("Ys correct>>", list(np.allclose(*x) for x in zip(compYs, Ys)))
        compPs = np.array([ Y @ P_ for Y, P_ in zip(compYs, Ps_) ])
        Ss = np.array([ R @ Z_ @ Y_.T @ R.T for Z_, Y_ in zip(Zs_, Ys_) ])

        S_ = np.zeros((L, n))
        Ms_ = np.zeros((L, n, n))

        for l in range(L):
            for a in range(n):
                S_[l,a] = Ss[a,l,l]
                for j in range(n):
                    Ms_[l, a, j] = compPs[j, l, a] / S_[l, a]

        if L == 2:
            min01, min02 = np.min(np.abs(S_ - mixture.S)), np.min(np.abs(S_[[1,0]] - mixture.S))
            perm = [0,1] if min01 < min02 else [1,0]
        else:
            perm = range(L)
        allS_.append(S_[perm])
        allMs_.append(Ms_[perm])

        m = Mixture(S_[perm], Ms_[perm])
        m.print()
        print(np.isclose(S_[perm], mixture.S, rtol=0, atol=1e-4).astype(int))
        print(np.isclose(Ms_[perm], mixture.Ms, rtol=0, atol=1e-4).astype(int))
    print(55 * "=")

    # import code; code.interact(local=locals())

    S_ = np.median(allS_, axis=0)
    Ms_ = np.median(allMs_, axis=0)

    # S_ = np.clip(np.real(S_), 0, 1)
    # Ms_ = np.clip(np.real(Ms_), 0, 1)
    # print("learned mixture:")
    # print(S_)
    # print(Ms_)
    print(f"below 0: {-np.sum(Ms_[Ms_ < 0])}, above 1: {np.sum(np.abs(1 - Ms_[Ms_ > 1]))}")
    S_ = np.abs(S_)
    Ms_ = np.abs(Ms_)
    learned_mixture = Mixture(S_, Ms_)
    # learned_mixture.normalize()
    return learned_mixture


def find_connected_components(edge_list):
    es = {}
    for e in list(edge_list):
        for i, j in [e, e[::-1]]:
            if i not in es:
                es[i] = []
            if j not in es[i]:
                es[i].append(j)
    cc = []
    while es:
        for w in es.keys(): break
        component = [w]
        i = 0
        while i < len(component):
            v = component[i]
            for u in es[v]:
                if u not in component:
                    component.append(u)
            del es[v]
            i += 1
        cc.append(component)
    return cc


def svd_learn3(sample, n, L=None, sv_threshold=0.05, verbose=None):
    # assert(len(sample.trails) == n**sample.t_len)
    Os = np.moveaxis(sample.all_trail_probs(), 1, 0)

    svds = [ np.linalg.svd(Os[j], full_matrices=True) for j in range(n) ]

    if verbose:
        for i, (_, s, _) in enumerate(svds):
            print(f"{i}: {s[:L+1]} ...")

    if L is None:
        above_thresh = [ np.sum(s / s[0] > sv_threshold) for _, s, _ in svds ]
        L = int(np.median(above_thresh))
        if verbose is not None:
            for (_, s, _), t in zip(svds, above_thresh):
                print(s / s[0], t)
            print("Guessed L={}".format(L))

    Ps_ = np.zeros((n, L, n))
    Qs_ = np.zeros((n, L, n))
    for j, (u, s, vh) in enumerate(svds):
        Ps_[j, 0:min(n,L), :] = u[:, 0:L].T
        Qs_[j, 0:min(n,L), :] = (np.diag(s) @ (vh))[0:L, :]

    A = np.zeros((2 * n * L, n**2))
    for j in range(n):
        A[L*j:L*(j+1), n*j:n*(j+1)] = Ps_[j]
        A[L*(n+j):L*(n+j+1), j+n*(np.arange(n))] = -Qs_[j]

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    print("singular values of A:", *[f"{x:.5f}" for x in s ])
    k = sum(s < 1e-5)
    print(f"{k} are about 0 (ideal is {L})")
    B = vh[-k:]
    print("shape of co-kernel is", B.shape)
    Bre = np.moveaxis(B.reshape((k, L, 2*n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    Ps = [ Zs_[i] @ Ys_[i].T for i in range(n) ]
    PsInv = [ np.linalg.pinv(P) for P in Ps ]

    for i, P in enumerate(Ps):
        print(f"-{i}-----------------------")
        print(P)
        print(np.linalg.matrix_rank(P))
        s, _ = np.linalg.eig(P)
        print(s)
        print("-------------------------")

    Rs_ = {}
    Es_ = {}
    for i, j in itertools.combinations(range(n), 2):
        print(f"-{i}-{j}------------------")
        X = np.linalg.pinv(Zs_[i] @ Ys_[i].T) @ (Zs_[j] @ Ys_[j].T)
        s, R_ = np.linalg.eig(X)
        mask = s > 1e-5
        E_ = R_[:,mask]
        if sum(mask) == L:
            Rs_[(i,j)] = R_
            Es_[(i,j)] = E_
            # import code; code.interact(local=locals())
        # print(R_.shape)
        # print(np.round(R_, 3))
        print(np.round(E_, 3))
        # print(R2_ @ np.diag(s[mask]) @ R2_.T)
        # print(R_ @ np.diag(s) @ np.linalg.inv(R_))
        print(np.linalg.norm(E_ @ np.diag(s[mask]) @ np.linalg.pinv(E_) - X))
        # print(X)
        print(s)
        print(np.absolute(s) > 1e-5)
        print("----------------------")

    EsGroups = []
    for e, E_ in Es_.items():
        for ix, (E2_, _) in enumerate(EsGroups):
            tol = 1e-2
            if np.linalg.norm(E_ - E2_) < tol:
                EsGroups[ix][1].append(e)
                break
        else:
            ix = len(EsGroups)
            EsGroups.append((E_, [e]))

    for E_, es in EsGroups:
        print(E_)
        print(es)
        print("==========================")

    cc = find_connected_components(Rs_.keys())
    print(f"connected components are {cc}")

    S_ = np.zeros((L, n))
    Ms_ = np.zeros((L, n, n))

    for component in cc:
        print(f"--- reconstructing component {component} ------------------")
        i, j = component[0], component[1]
        i, j = min(i,j), max(i,j)
        R_ = Rs_[(i,j)]
        print(R_)
        print(f"lstsq(({R_.T.shape} @ {Ys_[i].shape} @ {Ps_[i].shape}).T, {Os[i].shape} @ {np.ones(n).shape})")
        d, residuals, _, _ = np.linalg.lstsq((R_.T @ Ys_[i] @ Ps_[i]).T, Os[i] @ np.ones(n), rcond=None)
        print("d =", d, f"({residuals})")

        mask = np.abs(d) > 1e-5
        R = (np.diag(d) @ R_.T)[mask]
        print("R=")
        print(R)
        Ys = R @ Ys_

        Ps = np.array([ Y @ P_ for Y, P_ in zip(Ys, Ps_) ])
        Ss = np.array([ R @ Z_ @ Y_.T @ R.T for Z_, Y_ in zip(Zs_, Ys_) ])

        cS_ = np.zeros((L, n))
        cMs_ = np.zeros((L, n, n))

        for l in range(L):
            for i in range(n):
                if i in component: S_[l,i] = Ss[i,l,l]
                cS_[l,i] = Ss[i,l,l]
                for j in range(n):
                    if i in component: Ms_[l,i,j] = Ps[j,l,i] / S_[l,i]
                    cMs_[l, i, j] = Ps[j, l, i] / S_[l, i]

        print("S, M =")
        m = Mixture(cS_, cMs_)
        m.print()
    print("---------------------------------------------------------------")

    # S_ = np.clip(np.real(S_), 0, 1)
    # Ms_ = np.clip(np.real(Ms_), 0, 1)
    # print("learned mixture:")
    # print(S_)
    # print(Ms_)
    print(f"below 0: {-np.sum(Ms_[Ms_ < 0])}, above 1: {np.sum(np.abs(1 - Ms_[Ms_ > 1]))}")
    S_ = np.abs(S_)
    Ms_ = np.abs(Ms_)
    learned_mixture = Mixture(S_, Ms_)
    # learned_mixture.normalize()
    return learned_mixture


def svd_learn2(sample, n, L=None, sv_threshold=0.05, verbose=None):
    # assert(len(sample.trails) == n**sample.t_len)
    Os = np.moveaxis(sample.all_trail_probs(), 1, 0)

    svds = [ np.linalg.svd(Os[j], full_matrices=True) for j in range(n) ]

    if verbose:
        for i, (_, s, _) in enumerate(svds):
            print(f"{i}: {s[:L+1]} ...")

    if L is None:
        above_thresh = [ np.sum(s / s[0] > sv_threshold) for _, s, _ in svds ]
        L = int(np.median(above_thresh))
        if verbose is not None:
            for (_, s, _), t in zip(svds, above_thresh):
                print(s / s[0], t)
            print("Guessed L={}".format(L))

    Ps_ = np.zeros((n, L, n))
    Qs_ = np.zeros((n, L, n))
    for j, (u, s, vh) in enumerate(svds):
        Ps_[j, 0:min(n,L), :] = u[:, 0:L].T
        Qs_[j, 0:min(n,L), :] = (np.diag(s) @ (vh))[0:L, :]

    A = np.zeros((2 * n * L, n**2))
    for j in range(n):
        A[L*j:L*(j+1), n*j:n*(j+1)] = Ps_[j]
        A[L*(n+j):L*(n+j+1), j+n*(np.arange(n))] = -Qs_[j]

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    print("singular values of A:", *[f"{x:.5f}" for x in s ])
    k = sum(s < 1e-5)
    print(f"{k} are about 0 (ideal is {L})")
    B = vh[-k:]
    print("shape of co-kernel is", B.shape)
    Bre = np.moveaxis(B.reshape((k, L, 2*n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    print([ np.linalg.matrix_rank(Zs_[j] @ Ys_[j].T) for j in range(n) ])
    print([ np.linalg.svd(Zs_[j] @ Ys_[j].T)[1] for j in range(n) ])
    # Xs = [ np.linalg.inv(Zs_[j] @ Ys_[j].T) @ (Zs_[j+1] @ Ys_[j+1].T) for j in range(n-1) ]
    # Xs = [ np.linalg.pinv(Zs_[j] @ Ys_[j].T) @ (Zs_[j+1] @ Ys_[j+1].T) for j in range(n-1) ]
    # X = np.sum(Xs, axis=0)
    print(Zs_[0].shape)

    """
    from sim_diag import jacobi_angles
    R, L, err = jacobi_angles(np.linalg.pinv(Zs_[0] @ Ys_[0].T) @ (Zs_[1] @ Ys_[1].T),
                              np.linalg.pinv(Zs_[2] @ Ys_[2].T) @ (Zs_[3] @ Ys_[3].T))
    print(R)
    print(L)
    print(err)
    """

    Ps = [ Zs_[i] @ Ys_[i].T for i in range(n) ]
    PsInv = [ np.linalg.pinv(P) for P in Ps ]

    for i, P in enumerate(Ps):
        print(f"-{i}-----------------------")
        print(P)
        # v, w = np.linalg.eig(Ps[0])
        # R_ = np.linalg.pinv(w @ np.diag(np.sqrt(v)) @ np.linalg.pinv(w))
        print("-------------------------")

    R_bak = None
    for i, j in itertools.product(range(n), repeat=2):
        if i >= j:
            continue
        print(f"-{i}-{j}------------------")
        X = np.linalg.pinv(Zs_[i] @ Ys_[i].T) @ (Zs_[j] @ Ys_[j].T)
        s, R_ = np.linalg.eig(X)
        mask = s > 1e-5
        R2_ = R_[:,mask]
        if sum(mask) == 2 and R_bak is None:
            R_bak = R_
            # import code; code.interact(local=locals())
        # print(R_.shape)
        # print(np.round(R_, 3))
        print(np.round(R2_, 3))
        # print(R2_ @ np.diag(s[mask]) @ R2_.T)
        # print(R_ @ np.diag(s) @ np.linalg.inv(R_))
        print(np.linalg.norm(R2_ @ np.diag(s[mask]) @ np.linalg.pinv(R2_) - X))
        # print(X)
        print(s)
        print(np.absolute(s) > 1e-5)
        print("----------------------")
    R_ = R_bak
    print(R_)
    print(f"lstsq(({R_.T.shape} @ {Ys_[0].shape} @ {Ps_[0].shape}).T, {Os[0].shape} @ {np.ones(n).shape})")
    d, residuals, _, _ = np.linalg.lstsq((R_.T @ Ys_[0] @ Ps_[0]).T, Os[0] @ np.ones(n), rcond=None)
    print("d =", d, f"({residuals})")

    mask = np.abs(d) > 1e-5
    R = (np.diag(d) @ R_.T)[mask]
    print("R=")
    print(R)
    Ys = R @ Ys_

    Ps = np.array([ Y @ P_ for Y, P_ in zip(Ys, Ps_) ])
    Ss = np.array([ R @ Z_ @ Y_.T @ R.T for Z_, Y_ in zip(Zs_, Ys_) ])
    # print(Ss)

    S_ = np.zeros((L, n))
    Ms_ = np.zeros((L, n, n))
    for l in range(L):
        for i in range(n):
            S_[l,i] = Ss[i,l,l]
            for j in range(n):
                Ms_[l,i,j] = Ps[j,l,i] / S_[l,i]

    # S_ = np.clip(np.real(S_), 0, 1)
    # Ms_ = np.clip(np.real(Ms_), 0, 1)
    print("learned mixture:")
    print(S_)
    print(Ms_)
    print(f"below 0: {-np.sum(Ms_[Ms_ < 0])}, above 1: {np.sum(np.abs(1 - Ms_[Ms_ > 1]))}")
    S_ = np.abs(S_)
    Ms_ = np.abs(Ms_)
    learned_mixture = Mixture(S_, Ms_)
    # learned_mixture.normalize()
    return learned_mixture


def svd_learn(sample, n, L=None, sv_threshold=0.05, verbose=None, stats={}, mixture=None, Os=None):
    # assert(len(sample.trails) == n**sample.t_len)
    Os = np.moveaxis(sample.all_trail_probs(), 1, 0) if Os is None else Os

    svds = [ np.linalg.svd(Os[j], full_matrices=True) for j in range(n) ]

    if verbose:
        for i, (_, s, _) in enumerate(svds):
            print(f"{i}: {s[:L+1]} ...")

    if L is None:
        above_thresh = [ np.sum(s / s[0] > sv_threshold) for _, s, _ in svds ]
        L = int(np.median(above_thresh))
        if verbose is not None:
            for (_, s, _), t in zip(svds, above_thresh):
                print(s / s[0], t)
            print("Guessed L={}".format(L))

    Ps_ = np.zeros((n, L, n))
    Qs_ = np.zeros((n, L, n))
    for j, (u, s, vh) in enumerate(svds):
        Ps_[j, 0:min(n,L), :] = u[:, 0:L].T
        Qs_[j, 0:min(n,L), :] = (np.diag(s) @ (vh))[0:L, :]

    A = np.zeros((2 * n * L, n**2))
    for j in range(n):
        A[L*j:L*(j+1), n*j:n*(j+1)] = Ps_[j]
        A[L*(n+j):L*(n+j+1), j+n*(np.arange(n))] = -Qs_[j]

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    small = list(s < 1e-5)
    if True in small:
        fst = small.index(True)
        if verbose: print(2*L*n - fst, L, s[[fst-1, fst]])
    B = vh[-L:]
    Bre = np.moveaxis(B.reshape((L, L, 2*n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    if verbose:
        print([ np.linalg.matrix_rank(Zs_[j] @ Ys_[j].T) for j in range(n) ])
        print([ np.linalg.svd(Zs_[j] @ Ys_[j].T)[1] for j in range(n) ])
    # Xs = [ np.linalg.inv(Zs_[j] @ Ys_[j].T) @ (Zs_[j+1] @ Ys_[j+1].T) for j in range(n-1) ]
    Xs = [ np.linalg.pinv(Zs_[j] @ Ys_[j].T) @ (Zs_[j+1] @ Ys_[j+1].T) for j in range(n-1) ]
    X = np.sum(Xs, axis=0)
    # X = np.linalg.inv(Zs_[0] @ Ys_[0].T) @ (Zs_[1] @ Ys_[1].T)
    _, R_ = np.linalg.eig(X)
    d, _, _, _ = np.linalg.lstsq((R_.T @ Ys_[0] @ Ps_[0]).T, Os[0] @ np.ones(n), rcond=None)
    # maybe average over d, too?

    R = np.diag(d) @ R_.T
    Ys = R @ Ys_

    Ps = np.array([ Y @ P_ for Y, P_ in zip(Ys, Ps_) ])
    Ss = np.array([ R @ Z_ @ Y_.T @ R.T for Z_, Y_ in zip(Zs_, Ys_) ])
    # print(Ss)

    S_ = np.zeros((L, n))
    Ms_ = np.zeros((L, n, n))
    for l in range(L):
        for i in range(n):
            S_[l,i] = Ss[i,l,l]
            for j in range(n):
                # only good if warning "Casting complex values to real discards the imaginary part" occurs here:
                Ms_[l,i,j] = Ps[j,l,i] / S_[l,i]

    # S_ = np.clip(np.real(S_), 0, 1)
    # Ms_ = np.clip(np.real(Ms_), 0, 1)
    S_ = np.abs(S_)
    Ms_ = np.abs(Ms_)
    learned_mixture = Mixture(S_, Ms_)
    if verbose: print(learned_mixture)
    learned_mixture.normalize()
    return learned_mixture


def em_exponential_search(sample, n, min_tv_dist=0.01, max_iter=100, verbose=None,
                          reuse=True):
    L = 1
    lower_L = L
    upper_L = None
    n_iter = 0
    mixtures = {}

    for n_iter in range(max_iter):
        init_mixture = None
        if reuse and lower_L in mixtures:
            lower_mixture = mixtures[lower_L]
            ixMs = np.random.randint(lower_L, size=L-lower_L)
            ixS = np.random.randint(lower_L, size=L-lower_L)
            Ms = np.stack([ M for M in lower_mixture.Ms ] + [ lower_mixture.Ms[ix] for ix in ixMs ])
            S = np.stack([ M for M in lower_mixture.S ] + [ lower_mixture.S[ix] for ix in ixS ])
            eps = 1e-5
            init_mixture = Mixture(S + eps * np.random.rand(*S.shape),
                                   Ms + eps * np.random.rand(*Ms.shape))
            init_mixture.normalize()

        learned_mixture = em_learn(sample, n, L, init_mixture=init_mixture, ll_stop=1e-10)
        mixtures[L] = learned_mixture
        learned_distribution = Distribution.from_mixture(learned_mixture, sample.t_len)
        tv_dist = sample.dist(learned_distribution)
        if verbose is not None:
            print("Iteration {}: L={}, tv_dist={}".format(n_iter+1, L, tv_dist))

        if upper_L is not None and upper_L == L:
            break

        if tv_dist > min_tv_dist:
            if upper_L is None:
                lower_L = L
                L = 2 * L
            else:
                lower_L = L
        else:
            upper_L = L
        if upper_L is not None:
            L = (lower_L + upper_L + 1) // 2
        n_iter += 1
    return learned_mixture


"""
def exponential_search(sample, n, learn, target_tv_dist=0.0001, max_iter=100):
    L = 1
    lower_L = L
    upper_L = None
    n_iter = 0
    while (upper_L is None or upper_L > L) and n_iter < max_iter:
        learned_mixture = learn(sample, n, L)
        learned_distribution = Distribution.from_mixture(learned_mixture, sample.t_len)
        # maybe repeat and use majority direction? Repeat until variation diminishes?
        tv_dist = sample.dist(learned_distribution)
        print("Iteration {}: L={}, tv_dist={}".format(n_iter+1, L, tv_dist))
        if tv_dist > target_tv_dist:
            if upper_L is None:
                lower_L = L
                L = 2 * L
            else:
                lower_L = L
        else:
            upper_L = L
        if upper_L is not None:
            L = (lower_L + upper_L + 1) // 2
        n_iter += 1
    return learned_mixture
"""


def em_learn(sample, n, L, max_iter=1000, ll_stop=1e-4, verbose=None, init_mixture=None, stats={}, mixture=None,
             write_stats=False):
    flat_mixture = (init_mixture or Mixture.random(n, L)).flat()
    flat_trails, trail_probs = sample.flat_trails()

    prev_lls = 0
    for n_iter in range(max_iter):
        lls = flat_trails @ np.log(flat_mixture + 1e-20).transpose()
        if ll_stop is not None and np.max(np.abs(prev_lls - lls)) < ll_stop: break
        prev_lls = lls

        raw_probs = np.exp(lls)
        cond_probs = raw_probs / np.sum(raw_probs, axis=1)[:, np.newaxis]
        cond_probs[np.any(np.isnan(cond_probs), axis=1)] = 1 / L

        flat_mixture = cond_probs.transpose() @ (flat_trails * trail_probs[:, np.newaxis])
        # normalize:
        flat_mixture[:, :n] /= np.sum(flat_mixture[:, :n])
        for i in range(n):
            rows = flat_mixture[:, (i+1)*n:(i+2)*n]
            rows[:] = rows / rows.sum(axis=1)[:, np.newaxis]
            rows[np.any(np.isnan(rows), axis=1)] = 1 / n

        if verbose is not None:
            learned_mixture = Mixture.from_flat(flat_mixture, n)
            learned_distribution = Distribution.from_mixture(learned_mixture, sample.t_len)
            print("Iteration {}: recovery_error={} tv_dist={}".format(
                n_iter + 1, verbose.recovery_error(learned_mixture) if isinstance(verbose, Mixture) else np.inf,
                learned_distribution.dist(sample)))

    if write_stats: stats["n_iter"] = n_iter
    return Mixture.from_flat(flat_mixture, n)


"""
def svd_learn(sample, n, L):
    assert(len(sample.trails) == n**sample.t_len)
    Os = np.moveaxis(sample.trail_probs, 1, 0)

    Ps_ = np.zeros((n, L, n))
    Qs_ = np.zeros((n, L, n))
    for j in range(n):
        u, s, vh = np.linalg.svd(Os[j], full_matrices=True)
        Ps_[j, 0:min(n,L), :] = u[:, 0:L].T
        Qs_[j, 0:min(n,L), :] = (np.diag(s) @ (vh))[0:L, :]

    A = np.zeros((2 * n * L, n**2))
    for j in range(n):
        A[L*j:L*(j+1), n*j:n*(j+1)] = Ps_[j]
        A[L*(n+j):L*(n+j+1), j+n*(np.arange(n))] = -Qs_[j]

    _, _, vh = np.linalg.svd(A.T, full_matrices=True)
    B = vh[-L:]
    Bre = np.moveaxis(B.reshape((L, L, 2*n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    X = np.linalg.inv(Zs_[0] @ Ys_[0].T) @ (Zs_[1] @ Ys_[1].T)
    _, R_ = np.linalg.eig(X)
    d, _, _, _ = np.linalg.lstsq((R_.T @ Ys_[0] @ Ps_[0]).T, Os[0] @ np.ones(n), rcond=None)

    R = np.diag(d) @ R_.T
    Ys = R @ Ys_

    Ps = np.array([ Y @ P_ for Y, P_ in zip(Ys, Ps_) ])
    Ss = np.array([ R @ Z_ @ Y_.T @ R.T for Z_, Y_ in zip(Zs_, Ys_) ])
    # print(Ss)

    S_ = np.zeros((L, n))
    Ms_ = np.zeros((L, n, n))
    for l in range(L):
        for i in range(n):
            S_[l,i] = Ss[i,l,l]
            for j in range(n):
                Ms_[l,i,j] = Ps[j,l,i] / S_[l,i]

    S_ = np.clip(np.real(S_), 0, 1)
    Ms_ = np.clip(np.real(Ms_), 0, 1)
    learned_mixture = Mixture(S_, Ms_)
    learned_mixture.normalize()
    return learned_mixture
"""






# n = 25
# L = 10

# n = 3
# L = 2

# n >= 2*L (!)

"""
Ms = [[[0.25, 0.25, 0.5],
       [0.5, 0.1, 0.4],
       [0.5, 0.25, 0.25]],
      [[0.5, 0.3, 0.2],
       [0.8, 0.1, 0.1],
       [0.4, 0.3, 0.3]]]
S = [[0.2, 0.25, 0.2],
     [0.2, 0.05, 0.1]]
"""

"""
n = 2
Ms = [[[0.5, 0.5],
       [0.5, 0.5]],
      [[0.8, 0.2],
       [0.8, 0.2]]]
S = [[0.25, 0.25],
     [0.25, 0.25]]
"""

# mixture = Mixture(np.array(S), np.array(Ms))

"""
res = []

for n_trial in range(100):
    mixture = Mixture.random(n, L)
    # sample = Distribution(mixture, 3).sample_probs(10000000)
    sample = Distribution.from_mixture(mixture, 3).sample(10000000)
    # sample = Distribution(mixture, 3).sample_artificial_probs(0.00001)
    # sample = Distribution(mixture, 3).trail_probs
    t = time.time()
    learned_mixture = svd_learn(sample, n, L)
    # learned_mixture = em_learn(sample, n, L, verbose=mixture)
    # learned_mixture = em_learn(sample, n, 3)
    # learned_mixture = exponential_search(sample, n, em_learn)
    # learned_mixture = svd_learn_unknown_L(sample, n)
    recovery_error = mixture.recovery_error(learned_mixture)
    if recovery_error < np.inf:
        res.append(recovery_error)
    print("Trial {}: time={} recovery_error={} (median={})".format(
        n_trial, time.time() - t, recovery_error, np.median(res)))
"""

# print("Recovery error:", mixture.recovery_error(learned_mixture))


"""
L = 2
mixture = Mixture.random(10, L)
distribution = Distribution.from_mixture(mixture, 3)
sample = distribution.sample(int(1e4))
f = SVDLearn(ingroup_n_pairs=10,
             ingroup_combine=SVDLearn.ingroup_combine_median,
             normalization=SVDLearn.normalization_abs_em)
learned_mixture = f(sample, L)

recovery_error = mixture.recovery_error(learned_mixture)
learned_distribution = Distribution.from_mixture(learned_mixture, 3)
tv_dist = distribution.dist(learned_distribution)

print(recovery_error)
print(tv_dist)
"""
