# Final Project - Aspect-Based Company Matching Platform: Cinder

**Course: 20597 - Natural Language Processing** <br>
**Group 2:** Marie Cieslar, Monika Kaczorowska, Arzum Karahan, Marta Laskowska, Javiera Rubio di Biase <br>
**Date:** May 26, 2025

Cinder is an NLP-powered company matching system built on employee reviews from Glassdoor. It enables candidates to discover companies that align with their workplace priorities—such as pay, work-life balance, culture, and growth.

The project leverages aspect-based sentiment analysis, Bayesian shrinkage, and a scoring-based matching engine to build a personalized recommendation system. Results are visualized and deployed through an interactive Streamlit app. Summarized reviews are shown using Google's Gemini 2.0 Flash.

---

## Project Structure

| File / Folder                                   | Description                                                                                       |
|-------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `EDA.ipynb`                                     | Performs initial exploratory analysis on the raw Glassdoor review dataset before aspect detection. It explores data distribution, review counts, and prepares the dataset for downstream tasks.                                            |
| `cleaning and pyabsa.ipynb`                     | Handles text preprocessing and applies aspect-based sentiment analysis using PyABSA to extract sentiment scores for 13 workplace aspects.|
| `eda_company_aspects.ipynb`                     | Conducts firm-level analysis using the aspect score matrix. Visualizes patterns across companies, including clustering by aspect sentiment profiles.|
| `summarization_models.ipynb`                    | Extracts and processes “pros” and “cons” summaries from reviews for each firm using Google's Gemini 2.0 Flash, providing qualitative insights to accompany the numerical scores.|
| `build_company_matrix.py`                       | Creates shrunk aspect-rating matrix using Bayesian smoothing for use in the matching engine.                                  |
| `app.py`                                        | Streamlit application to rank user preferences and return top matching firms.|

---

## How to Run

1. **Set up environment**  
We recommend using Python 3.9+ and installing dependencies:  
    - `pandas`, `numpy`, `pathlib`, `itertools`, `collections`, `math`, `re`
    - `matplotlib`, `seaborn`, `mpl_toolkits.mplot3d`
    - `nltk`, `spacy`, `textblob`, `symspellpy`, `transformers`, `pyabsa`
    - `sklearn`, `scipy`, `torch`, `umap-learn`
    - **Note:** Additional libraries and dependencies are specified directly within each notebook (see the first code cells).

2. **Run the notebooks in order**  
Open in Jupyter Notebook or VS Code (with the Jupyter extension):
    - `EDA.ipynb` → perform exploratory data analysis 
    - `cleaning and pyabsa.ipynb` → extract aspect sentiments from review texts
    - `eda_company_aspects.ipynb` → cluster companies and explore aspects
    - `build_company_matrix.py` → generate the Bayesian-shrunk company-aspect matrix via: <pre><code>python build_company_matrix.py</code></pre>
    - `summarization_models.ipynb` → extract pros/cons summaries for firms
    - `app.py` → launch the Streamlit app via: <pre><code>streamlit run app.py</code></pre>

---

## Outputs & Results

- Aspect-Based Sentiment Extraction using PyABSA
- Bayesian Shrinkage to stabilize estimates for firms with fewer reviews
- Interactive Cinder app to personalize company matches
- Qualitative context through pros and cons summaries via Gemini
- 3D Clustering Analysis to reveal firm archetypes based on culture, growth, and people