"""
Convert the long Glassdoor aspect table into a dense (firm  × aspect) matrix
with **Bayesian-shrunk ratings**.  The shrinkage guards against noisy cases
where an aspect is mentioned only once or twice.

Formula
-------
    shrunk_rating = (avg_star * n + μ * α) / (n + α)

* μ (GLOBAL_MEAN) = 3.0  -> ‘neutral’ on the 1–5 sentiment scale.

TRIED BUT NOT KEPT (used grid search and checked rmse --> found out that alpha=0 has the lowest RMSE)?
* α (ALPHA)       = 5    -> we pretend every firm starts with five
  neutral reviews; after ~5 real mentions, data outweighs prior. (after checking the distribution, the median is found to be 5 that's why we set the alpha as 5)
"""
import pandas as pd
import numpy  as np
import joblib, pathlib

#hyper-parameters
GLOBAL_MEAN = 3.0            # μ  – centre of sentiment scale
ALPHA       = 0              # α  – prior strength (“virtual” reviews)
CSV_PATH    = "Final_company_aspect_matrix_with_counts.csv"
OUT_DIR     = pathlib.Path("model_artifacts")
OUT_DIR.mkdir(exist_ok=True)

#loading the long table (firm, aspect, avg_star_rating, n_mentions)
df = pd.read_csv(CSV_PATH)

#credibility-weight each cell via Bayesian shrinkage
df["shrunk_rating"] = (
    (df.avg_star_rating * df.n_mentions + GLOBAL_MEAN * ALPHA)
    / (df.n_mentions + ALPHA)
)

#pivoting to wide matrix: rows = firms, columns = aspects
#missing combos get μ = 3.0 --> if we assign 0, then we will be punishing them but rather we need to have a neutral view (open to discussions)
mat = (
    df.pivot(index="firm",
             columns="aspect",
             values="shrunk_rating")
      .fillna(GLOBAL_MEAN)
      .sort_index()
)

#persist artefacts for the Streamlit app
V       = mat.to_numpy(dtype=np.float32)   # company × aspect matrix
firms   = mat.index.to_numpy()            # firm names in same order
aspects = list(mat.columns)               # aspect list / column order

np.save (OUT_DIR / "company_matrix.npy", V)
joblib.dump(firms,   OUT_DIR / "firms.joblib")
joblib.dump(aspects, OUT_DIR / "aspects.joblib")

print(f"Saved matrix {V.shape} and metadata to {OUT_DIR.resolve()}")