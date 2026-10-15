# Advertising spend explorer

An interactive walkthrough of a classic regression problem: given TV, radio and newspaper advertising budgets, how well can different models predict sales, and where should the next dollar go?

**Live demo: https://ryanward.dev/advertising-spend-explorer/** (part of my [projects](https://ryanward.dev/projects/))

## What it shows

- Basic exploration of the advertising dataset (summary statistics, scatter plots).
- Training four scikit-learn regressors (linear regression, random forest, k-nearest neighbours, support vector regression) with an adjustable train/test split.
- Predicted vs actual sales and RMSE, so you can compare models.
- Making your own prediction from a budget you choose.

## How it runs

There is no server. The app is a [Streamlit](https://streamlit.io) script that runs entirely in your browser via [stlite](https://github.com/whitphx/stlite) (Streamlit on [Pyodide](https://pyodide.org), Python compiled to WebAssembly). `index.html` loads stlite from a CDN and mounts `streamlit_app.py`, so it is hosted as plain static files on GitHub Pages and costs nothing to run. The first load takes 10-30 seconds while Python and scikit-learn download.

To run it locally:

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

To run it as a normal Streamlit app instead: `pip install streamlit scikit-learn altair pandas && streamlit run streamlit_app.py`.

## History

Originally written in 2021 as a Streamlit Community Cloud app. This version was ported in 2026 to run in the browser, with these changes: `st.cache` replaced by `st.cache_data`, the CSV bundled instead of fetched at runtime, budget sliders limited to the range of the data (the original allowed predictions far outside it), a fixed seed for the random forest, and the fake "thinking" delay removed.

## Notes and limitations

- It is a teaching example on a small cross-sectional dataset (200 markets, no time dimension), so it says nothing about effects over time, such as carryover from past advertising.
- Support vector regression and k-nearest neighbours are used without feature scaling, which is part of why they perform worse here. That is deliberate for the comparison but worth knowing.

## Data and licence

`Advertising.csv` comes from [*An Introduction to Statistical Learning*](https://www.statlearning.com/) by James, Witten, Hastie and Tibshirani, and is not covered by this repository's licence. The code is released under the MIT licence (see `LICENSE`).
