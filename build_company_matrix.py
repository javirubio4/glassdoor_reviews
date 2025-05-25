"""
Convert the long Glassdoor aspect table into a dense (firm  × aspect) matrix
with **Bayesian-shrunk ratings**.  The shrinkage guards against noisy cases
where an aspect is mentioned in less than 5 reviews.

Formula:
    shrunk_rating = (avg_star * n + μ_aspect * α) / (n + α)

* μ_aspect = mean of the specific aspect across different countries
* α (ALPHA)       = 5    -> we pretend every firm starts with five
reviews where the mean is the μ_aspect; after ~5 real mentions, data outweighs prior. (after checking the distribution, the median is found to be 5 that's why we set the alpha as 5)
"""
import pandas as pd
import numpy as np
import joblib
import pathlib

# Hyperparameters
GLOBAL_MEAN = 3.0  # neutral reference point in 1-5 scale
ALPHA = 5          # shrinkage strength for low-mention cases
CSV_PATH = "Final_filtered_imputed_company_aspect_matrix_with_counts.csv"
OUT_DIR = pathlib.Path("model_artifacts_final_filtered")
OUT_DIR.mkdir(exist_ok=True)

#load the filtered table
df = pd.read_csv(CSV_PATH)

#calculate aspect-level means
aspect_means = df.groupby("aspect")["avg_star_rating"].mean().to_dict()

#define hybrid shrinkage function
def compute_shrunk_rating(row):
    n = row["n_mentions"]
    avg_star = row["avg_star_rating"]
    aspect_mean = aspect_means.get(row["aspect"], GLOBAL_MEAN) #in case of a fallback, global mean is used

    if n >= 5:
        return avg_star
    else:
        return (avg_star * n + aspect_mean * ALPHA) / (n + ALPHA)  #shrink noisy firm-aspect

#apply to each row
df["shrunk_rating"] = df.apply(compute_shrunk_rating, axis=1)

#pivot to firm × aspect matrix
mat = (
    df.pivot(index="firm", columns="aspect", values="shrunk_rating")
      .fillna(GLOBAL_MEAN) #there shouldn't be any missing values as we are using the imputed file
      .sort_index()
)

#persist artifacts for Streamlit app
V = mat.to_numpy(dtype=np.float32)
firms = mat.index.to_numpy()
aspects = list(mat.columns)

np.save(OUT_DIR / "company_matrix.npy", V)
joblib.dump(firms, OUT_DIR / "firms.joblib")
joblib.dump(aspects, OUT_DIR / "aspects.joblib")

print(f"Saved filtered matrix {V.shape} and metadata to {OUT_DIR.resolve()}")