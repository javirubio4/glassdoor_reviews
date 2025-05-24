"""
Cinder:
Vectorised recommender that matches a candidate’s ranked aspect
preferences to firms, using the pre-compressed matrix from
build_company_matrix.py.

Scoring rationale:
score(firm) = w · v_firm
    w      = normalised weights from the user  (sum = 1)
    v_firm = shrunk-rating vector of length K aspects
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pathlib


#load matrix artifacts
ART = pathlib.Path("model_artifacts_final_filtered")
V = np.load(ART / "company_matrix.npy")          # (firms, K aspects)
firms = joblib.load(ART / "firms.joblib")        # array of firm names
aspects = joblib.load(ART / "aspects.joblib")    # list of aspects
K = len(aspects)

#streamlit page config
st.set_page_config(
    page_title="Cinder (Filtered) | Company Matcher",
    page_icon="https://i.imgur.com/Jnp0v9A.png",
    layout="wide"
)


st.markdown("""
<div style="display: flex; align-items: center; gap: 1rem;">
    <img src="https://i.imgur.com/fAHhYvv.png" width="130"/>
    <h1 style="color:#3e9bea; margin: 0;"> – Company Matcher</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
h1 {
    color: #3e9bea;
    margin-top: 0;
    margin-bottom: 0.2rem;
}
div[data-testid="stDataFrame"] > div:first-child {
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

st.write("Pick up to five aspects and rank them to discover companies that match your vibe!")

#aspect selector
sel_cols = st.columns(5, gap="small")
choices = []
for r, col in enumerate(sel_cols, 1):
    with col:
        sel = st.selectbox(f"Rank {r}", [""] + aspects, key=f"rank_{r}")
        if sel:
            choices.append(sel)

#validation
if len(choices) != len(set(choices)):
    st.error("Duplicate aspects selected – please pick each once.")
    st.stop()
if not choices:
    st.stop()

st.markdown("---")

#compute scores
w = np.zeros(K, dtype=np.float32)
for idx, asp in enumerate(choices[::-1]):   # top rank → highest weight
    w[aspects.index(asp)] = idx + 1
w /= w.sum()

scores = V @ w
top_idx = scores.argsort()[::-1][:10]

rows = []
for pos, idx in enumerate(top_idx, 1):
    contrib = V[idx] * w
    rows.append({
        "#": pos,
        "firm": firms[idx],
        "score": scores[idx],
        **{a: contrib[aspects.index(a)] for a in choices}
    })

df_out = pd.DataFrame(rows).reset_index(drop=True).set_index("#")
fmt = {"score": "{:.3f}", **{a: "{:.2f}" for a in choices}}

#display top 10 matching companies
st.markdown("### Best-matching companies")
st.dataframe(df_out.style
             .format(fmt)
             .background_gradient(cmap="Greens", subset=["score"])
             .background_gradient(cmap="Blues", subset=choices),
             use_container_width=True, height=420)

#download results
st.download_button("⬇️ Download results as CSV",
                   data=df_out.to_csv().encode(),
                   file_name="cinder_top10_filtered.csv",
                   mime="text/csv")

#load company summaries
summary_path = "company_merged_summaries.csv"
df_summaries = pd.read_csv(summary_path)

#get top 3 firms and display summaries
top_3_firms = df_out["firm"].head(3).tolist()
top_3_summaries = (df_summaries.set_index("firm").loc[top_3_firms].reset_index())

st.markdown("### 📝 Summaries of Top 3 Companies")

for _, row in top_3_summaries.iterrows():
    st.markdown(f"#### {row['firm']}")

    st.markdown("**✅ Pros**")
    st.markdown(f"- **Summary:** {row['pros_summary']}")

    st.markdown("**⚠️ Cons**")
    st.markdown(f"- **Summary:** {row['cons_summary']}")
    

    st.markdown("---")
