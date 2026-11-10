import os, glob
import numpy as np, pandas as pd
R = "results/certification"; L = []
groups = {}
for f in sorted(glob.glob(os.path.join(R, "bench_*__*.csv"))):
    groups.setdefault(os.path.basename(f)[6:-4].split("__")[0], []).append(f)
fam = lambda d: "sim" if d.startswith("sim") else ("covid" if d.startswith("pbmc_covid") else ("lupus" if d.startswith("pbmc_lupus") else ("pbmc" if d.startswith("pbmc") else d)))
for tag, fs in groups.items():
    b = pd.concat([pd.read_csv(f) for f in fs]); b["family"] = b.dataset.map(fam)
    keep = b.method.isin(["fixed_128", "fixed_256", "fixed_512", "seq_iid", "seq_strat_nbr", "seq_sEB_n", "seq_sEB_b", "seq_sEB_r", "seq_sEB_nb", "seq_sEB_br"])
    g = b[keep].groupby(["family", "method"])[["decided", "error", "mean_runs", "median_runs"]].mean().round(3)
    L.append(f"=== {tag}\n" + g.to_string())
    ps = [f.replace("bench_", "perclaim_").replace(".csv", ".csv.gz") for f in fs]; ps = [p for p in ps if os.path.exists(p)]
    if not ps: continue
    pc = pd.concat([pd.read_csv(p) for p in ps]); pc["i"] = pc.groupby(["dataset", "method", "rep"]).cumcount()
    base = pc[pc.method == "seq_iid"].set_index(["dataset", "rep", "i"])
    for k, m in [("nbr", "seq_strat_nbr")] + [(q, f"seq_sEB_{q}") for q in ["n", "b", "r", "nb", "br"]]:
        if f"S_{k}" not in pc or (pc.method == m).sum() == 0: continue
        cv = pc[pc.method == m].set_index(["dataset", "rep", "i"])
        j = base[["t", f"S_{k}", "s"]].join(cv[["t"]], rsuffix="_cv")
        both = (j.t <= 1024) & (j.t_cv <= 1024) & (j.t > 3)
        q = pd.DataFrame({"S": j[f"S_{k}"][both], "ratio": (j.t_cv / j.t)[both]})
        q["Sbin"] = pd.cut(q.S, [-0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
        t = q.groupby("Sbin", observed=True).agg(n=("ratio", "size"), observed_runs_ratio=("ratio", "median"), mean_S=("S", "mean"))
        t["predicted_1_minus_S"] = 1 - t.mean_S
        rho = np.corrcoef(q.S, np.log(q.ratio))[0, 1] if len(q) > 10 else np.nan
        L.append(f"  [{m}] runs ratio vs theory (1 - S_closed); corr(S, log ratio) = {rho:.3f}; overall median ratio {q.ratio.median():.3f}\n" + t.round(3).to_string())
open("results/summary_cert.txt", "w").write("\n".join(L)); print("\n".join(L))
