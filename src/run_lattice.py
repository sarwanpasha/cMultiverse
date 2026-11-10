"""Run every pipeline sharing one (nhvg, norm) prefix. Task id t in [0, len(nhvg)*len(norm)).
(ncomp, k) units run in parallel worker processes (Leiden is single-threaded).
Usage: python run_lattice.py --name pbmc_s0 --task 7 [--quick] [--jobs 8]
"""
import argparse, os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import (RES, load_counts, levels_for, normalize, select_hvg, batch_matrix, harmony, scvi_embed, LEIDEN_SEEDS)


def unit(E, k, res_list, seeds):
    os.environ["NUMBA_NUM_THREADS"] = "1"; os.environ["OMP_NUM_THREADS"] = "1"
    import anndata as ad, scanpy as sc
    A = ad.AnnData(np.zeros((E.shape[0], 1), dtype=np.float32)); A.obsm["X_emb"] = E
    sc.pp.neighbors(A, n_neighbors=k, use_rep="X_emb", random_state=0)
    out = []
    for r in res_list:
        per_seed = []
        for s in seeds:
            sc.tl.leiden(A, resolution=r, flavor="igraph", n_iterations=2, directed=False, random_state=s, key_added="cl")
            per_seed.append(A.obs["cl"].astype(int).values.astype(np.int16))
        out.append((r, np.stack(per_seed)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--only_batch", default=None); ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()
    from joblib import Parallel, delayed
    from sklearn.decomposition import PCA
    t0 = time.time()
    X, obs, genes, meta = load_counts(a.name)
    lv = levels_for(meta)
    if a.quick:
        lv.update(ncomp=[10, 30], k=[15, 50], res=[0.5, 1.2], batch=lv["batch"][:2])
    if a.only_batch: lv["batch"] = [a.only_batch]
    nn, nm = len(lv["nhvg"]), len(lv["norm"])
    nhvg = lv["nhvg"][a.task // nm]; norm = lv["norm"][a.task % nm]
    out_dir = os.path.join(RES, a.name, "lattice" + ("_quick" if a.quick else "") + ("_" + a.only_batch if a.only_batch else "")); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"task_{a.task:03d}.npz")
    if os.path.exists(out):
        print("exists", out); sys.exit(0)
    print(f"[{a.name}] task {a.task}: nhvg={nhvg} norm={norm} cells={X.shape[0]} genes={X.shape[1]} jobs={a.jobs}", flush=True)
    Z = normalize(X, norm)
    hvg = select_hvg(Z, norm, min(nhvg, Z.shape[1]))
    Xh = np.ascontiguousarray(Z[:, hvg]); del Z
    rows, labs, pca_var = [], [], {}
    nmax = int(min(max(lv["ncomp"]), Xh.shape[1] - 1, Xh.shape[0] - 1))
    with Parallel(n_jobs=a.jobs, backend="loky") as par:
        for b in lv["batch"]:
            tb = time.time()
            embs = {}
            if b == "scvi":
                Xcnt = X[:, hvg].toarray().astype(np.float32)
                for nc in lv["ncomp"]:
                    embs[nc] = scvi_embed(Xcnt, obs, nc, threads=a.jobs); print(f"    scvi n_latent={nc} t={time.time()-t0:.0f}s", flush=True)
            else:
                M = batch_matrix(Xh, b, obs)
                pca = PCA(n_components=nmax, svd_solver="randomized", random_state=0)
                pcs = pca.fit_transform(M).astype(np.float32); del M
                pca_var[b] = pca.explained_variance_ratio_.tolist()
                for nc in lv["ncomp"]:
                    E = np.ascontiguousarray(pcs[:, :min(nc, nmax)])
                    embs[nc] = harmony(E, obs) if b == "harmony" else E
            units = [(nc, k) for nc in lv["ncomp"] for k in lv["k"]]
            results = par(delayed(unit)(embs[nc], k, lv["res"], LEIDEN_SEEDS) for nc, k in units)
            for (nc, k), res_out in zip(units, results):
                for r, L in res_out:
                    rows.append((nhvg, norm, nc, b, k, r)); labs.append(L)
            print(f"  batch={b} done in {time.time()-tb:.0f}s  (pipelines so far {len(rows)})", flush=True)
    labs = np.stack(labs)
    np.savez_compressed(out, labels=labs,
        nhvg=np.array([r[0] for r in rows]), norm=np.array([r[1] for r in rows]), ncomp=np.array([r[2] for r in rows]),
        batch=np.array([r[3] for r in rows]), k=np.array([r[4] for r in rows]), res=np.array([r[5] for r in rows]),
        hvg=hvg.astype(np.int32), pca_var=json.dumps(pca_var))
    print(f"saved {out}  {labs.shape}  total {time.time()-t0:.0f}s", flush=True)
