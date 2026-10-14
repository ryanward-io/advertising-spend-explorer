"""How should I spend my advertising dollars?

An interactive regression walkthrough on the advertising dataset from
"An Introduction to Statistical Learning". Runs entirely in the browser via
stlite (Streamlit on Pyodide); see index.html.

Originally written in 2021, ported and tidied in 2026.
"""
import altair as alt
import pandas as pd
import streamlit as st
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

st.set_page_config(page_title="How should I spend my advertising dollars?", layout="centered")

st.write("# How should I spend my advertising dollars?")
st.write("## A data science exploration by Ryan Ward")

st.write(
    """
Marketers often have trouble predicting in advance where to get the best value for their
limited budgets.

Data can help here, and it is easier than you think.
"""
)

st.write(
    """
For this exploration, we'll use an example advertising dataset from a terrific book,
*An Introduction to Statistical Learning*.

It contains the advertising budgets (in $'000) for a company in 200 different markets and
the number of sales of a product in each (in thousands of units).

We'll start with a brief exploration before conducting a type of machine learning called
regression. We can then make some predictions.
"""
)

st.write("## Into the analysis!")


@st.cache_data
def load_data():
    data = pd.read_csv("Advertising.csv", index_col=0)
    data.columns = ["TV", "Radio", "Newspaper", "Sales"]
    return data


data = load_data()

if st.checkbox("Show raw data"):
    st.write("### Raw data")
    st.write(data)

st.write("### Summary statistics")
st.dataframe(data.describe())

st.write(
    """
This gives us important information such as:

* **number of rows** in the table (200).
* **central tendencies** of the means and medians (50%). These are reasonably similar,
  which might indicate we are working with non-skewed data.
* **min and max** for each column can also be interesting to look into as they may
  contain outliers.
"""
)

st.write(
    "It can also be valuable to explore relationships between the features "
    "(TV, radio, newspaper) and the target (sales)."
)


def plot_scatter(x, y):
    chart = (
        alt.Chart(data)
        .mark_circle(size=60)
        .encode(x=x, y=y, tooltip=["TV", "Radio", "Newspaper", "Sales"])
        .properties(title=f"{x} budget vs {y}")
    )
    st.altair_chart(chart, use_container_width=True)


plot_scatter("TV", "Sales")
plot_scatter("Radio", "Sales")
plot_scatter("Newspaper", "Sales")

st.write(
    """
#### What do the charts tell us?

* **TV budget** _increases reliably_ with sales but plateaus at the higher values
* **Radio budget** also increases with sales but _not as strongly_ as TV
* **Newspaper budget** shows the _weakest relationship_ with sales

We can explore this further with a type of machine learning called regression.
"""
)

st.write("## So let's create a model!")

st.write("1. First we need to split the data into training and testing data.")
st.write(
    "We'll train the model on the training data and validate its performance "
    "on the testing data."
)

test_size = st.slider(
    "Choose the size of your test data (e.g. 0.30 represents a 30% test size)",
    0.1,
    0.4,
    0.3,
)

models = {
    "Linear Regression (OLS)": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "K Neighbours": KNeighborsRegressor(),
    "Support Vector Regression": SVR(),
}

st.write("2. Now let's choose a model to train.")
st.write(
    "It can be tricky to know which models to try but there is no harm "
    "in trying multiple different models and comparing results."
)

model_name = st.selectbox("What model do you want to try?", list(models))

y = data["Sales"]
X = data.drop("Sales", axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

reg = models[model_name]
reg.fit(X_train, y_train)

evaluation = pd.DataFrame({"Predictions": reg.predict(X_test), "Actuals": y_test})

points = (
    alt.Chart(evaluation)
    .mark_circle(size=60)
    .encode(alt.X("Predictions"), alt.Y("Actuals"))
    .properties(title="Performance of predicted vs actual sales")
)
st.altair_chart(
    points + points.transform_regression("Predictions", "Actuals").mark_line(),
    use_container_width=True,
)

rmse = round(
    metrics.mean_squared_error(evaluation["Actuals"], evaluation["Predictions"]) ** 0.5, 2
)

st.write("### How do we know if our model performed well?")
st.write(
    f"The square root of the mean squared error (RMSE) is one metric we can use to "
    f"measure how accurate our predictions are. Here, the RMSE was {rmse}.\n\n"
    f"This means that in general the sales prediction will be out by about "
    f"{int(rmse * 1000):,} units."
)
st.write("#### Is this good?")
st.write(
    "It depends! For some medical diagnosis model you might need to be far more accurate, "
    "but in a business context it might be fine."
)

st.write("### Making your own predictions")
st.write(
    "Now for the interesting bit: entering your own marketing budget and predicting sales.\n\n"
    "Try entering your budget for TV, radio and newspapers to see what the model predicts."
)
st.caption(
    "The sliders only cover the range of budgets in the data. Predictions outside that range "
    "would be extrapolation, which these models handle badly."
)


def budget_slider(label, column):
    low, high = float(data[column].min()), float(data[column].max())
    return st.slider(f"{label} budget (in $'000)", low, high, float(data[column].median()))


tv_value = budget_slider("TV", "TV")
radio_value = budget_slider("Radio", "Radio")
newspaper_value = budget_slider("Newspaper", "Newspaper")

if st.button("How many things am I going to sell?"):
    single_prediction = reg.predict(
        pd.DataFrame([[tv_value, radio_value, newspaper_value]], columns=X.columns)
    )
    st.success("Prediction complete")
    st.write(
        f"The model predicts you'll sell about **{single_prediction[0]:.1f}k units** from a "
        f"combined advertising budget of TV: \\${tv_value:.0f}k, Radio: \\${radio_value:.0f}k and "
        f"Newspaper: \\${newspaper_value:.0f}k."
    )
    st.write(
        "Tinker with the inputs and see if you can find an optimal budget: "
        "making the most sales for the least cost!"
    )
    st.write("Questions? Find me at [ryanward.dev](https://ryanward.dev).")
