import streamlit as st
import pandas as pd

# Load data
matrix = pd.read_csv("company_aspect_matrix.csv")
wide = matrix.pivot(index='firm', columns='aspect', values='avg_star_rating').fillna(0)
aspects = sorted(wide.columns.tolist())

# UI
st.title("Find Your Ideal Company")
st.markdown("### Rank the most important aspects to you:")

# Rank input
ranked_aspects = []
cols = st.columns(5)
for i in range(5):
    with cols[i]:
        choice = st.selectbox(f"{i+1}:", [""] + aspects, key=f"aspect_{i}")
        ranked_aspects.append(choice)

# Remove blanks
ranked_aspects = [a for a in ranked_aspects if a]

# Check for duplicates
duplicates = len(ranked_aspects) != len(set(ranked_aspects))

if duplicates:
    st.error("⚠️ Each aspect must be unique. Please don't select the same aspect more than once.")
elif st.button("Find Best Companies") and ranked_aspects:
    weights = [5, 4, 3, 2, 1][:len(ranked_aspects)]
    score_df = wide[ranked_aspects].copy()
    score_df["weighted_score"] = sum(score_df[aspect] * weight for aspect, weight in zip(ranked_aspects, weights))
    score_df = score_df.sort_values("weighted_score", ascending=False)

    def highlight_weights(row):
        return ['background-color: #f0f0f0' if col == 'weighted_score' else '' for col in row.index]

    st.markdown("### Best Matching Companies")
    st.dataframe(score_df.head(10).style.apply(highlight_weights, axis=1).format("{:.2f}"))
else:
    st.caption("Select at least one aspect to begin.")
