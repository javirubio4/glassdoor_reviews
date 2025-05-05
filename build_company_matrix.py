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
import numpy as np
import joblib
import pathlib

# Hyperparameters
GLOBAL_MEAN = 3.0  # μ – neutral sentiment
ALPHA = 5          # ⭐ Shrinkage strength for low-mention cases
CSV_PATH = "Extra_filtered_company_aspect_matrix.csv"
OUT_DIR = pathlib.Path("model_artifacts_filtered")  # Save separately
OUT_DIR.mkdir(exist_ok=True)

# Load the filtered table
df = pd.read_csv(CSV_PATH)

# --- Step 1: Calculate aspect-level means (market averages across all firms)
aspect_means = df.groupby("aspect")["avg_star_rating"].mean().to_dict()

# --- Step 2: Define hybrid shrinkage function
def compute_shrunk_rating(row):
    n = row["n_mentions"]
    avg_star = row["avg_star_rating"]
    aspect_mean = aspect_means.get(row["aspect"], GLOBAL_MEAN)

    if n >= 5:
        return avg_star  # Trust firm's own rating
    else:
        return (avg_star * n + aspect_mean * ALPHA) / (n + ALPHA)  # Shrink noisy firm-aspect

# --- Step 3: Apply to each row
df["shrunk_rating"] = df.apply(compute_shrunk_rating, axis=1)

# --- Step 4: Pivot to firm × aspect matrix
mat = (
    df.pivot(index="firm", columns="aspect", values="shrunk_rating")
      .fillna(GLOBAL_MEAN)  # Fill missing firm-aspect pairs neutrally
      .sort_index()
)

# --- Step 5: Persist artifacts for Streamlit app
V = mat.to_numpy(dtype=np.float32)
firms = mat.index.to_numpy()
aspects = list(mat.columns)

np.save(OUT_DIR / "company_matrix.npy", V)
joblib.dump(firms, OUT_DIR / "firms.joblib")
joblib.dump(aspects, OUT_DIR / "aspects.joblib")

print(f"✅ Saved filtered matrix {V.shape} and metadata to {OUT_DIR.resolve()}")