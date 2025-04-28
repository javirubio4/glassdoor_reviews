"""
Cinder demo (no Streamlit cache)

Vectorised recommender that matches a candidate’s ranked aspect
preferences to firms, using the pre-compressed matrix from
build_company_matrix.py.

Scoring rationale
-----------------
score(firm) = w · v_firm
    w      = normalised weights from the user  (sum = 1)
    v_firm = shrunk-rating vector of length K aspects
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib, pathlib

#artefact load
ART = pathlib.Path("model_artifacts")
V       = np.load (ART / "company_matrix.npy")           # (Firms, K)
firms   = joblib.load(ART / "firms.joblib")              # ndarray[str]
aspects = joblib.load(ART / "aspects.joblib")            # list[str]
K       = len(aspects)

#Streamlit page config
st.set_page_config(page_title="Cinder | Company Matcher",
                   page_icon="🔥", layout="wide")

#CSS polish
st.markdown("""
<style>
h1 { color:#ff6b00; margin-top:0; margin-bottom:0.2rem;}
div[data-testid="stDataFrame"] > div:first-child {
    border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.08);}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🔥 Cinder – Company-Candidate Matcher</h1>",
            unsafe_allow_html=True)
st.write("Pick up to five aspects and rank them from most to least important "
         "to discover companies that match your vibe.")

#selector bar
sel_cols = st.columns(5, gap="small")
choices  = []
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

#computing scores
w = np.zeros(K, dtype=np.float32)
for idx, asp in enumerate(choices[::-1]):       # top rank → highest weight
    w[aspects.index(asp)] = idx + 1
w /= w.sum()

scores  = V @ w
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

df_out = (pd.DataFrame(rows)
            .reset_index(drop=True)
            .set_index("#"))
fmt = {"score": "{:.3f}", **{a: "{:.2f}" for a in choices}}

#displaying
st.markdown("### Best-matching companies")
st.dataframe(df_out.style
             .format(fmt)
             .background_gradient(cmap="Greens", subset=["score"])
             .background_gradient(cmap="Blues", subset=choices),
             use_container_width=True, height=420)

st.download_button("⬇️ Download results as CSV",
                   data=df_out.to_csv().encode(),
                   file_name="cinder_top10.csv",
                   mime="text/csv")