import streamlit as st
import streamlit.components.v1 as components
import shap
from streamlit_shap import st_shap
import pickle
import pandas as pd

# should/has to be first command in script
st.set_page_config(page_title="Boston Housing model", layout="wide")


# load pickle file
# write as function so we can cache it
@st.experimental_memo
def load_model(pkl):
    return pickle.load(open(pkl, "rb"))


model = load_model("model_files/housing_catboost.pkl")

# write as a function so we can cache it
# use singleton here since there can only be one input dataset
@st.experimental_singleton
def load_data():
    return shap.datasets.boston()


X, y = load_data()

#### Streamlit app ####
"## Boston House Price model"
"""Adapted from [catboost example](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Catboost%20tutorial.html) in [SHAP documentation](https://shap.readthedocs.io/en/latest/index.html)"""
"Model last run: 2022-01-31 07:12"

## sidebar info ##
with st.sidebar.expander("Project Goals"):
    """
    1. To provide client with a model for Boston housing prices, based on the classic dataset from the paper **'Hedonic prices and the demand for clean air'**, J. Environ. Economics & Management, vol.5, 81-102, 1978'
    2. Explain how the model works using the Shapley Additive Explanations (SHAP) framework
    3. Get buy-in from relevant stakeholders, then use this model to predict which houses are good investment opportunities
    """

with st.sidebar.expander("How does the SHAP package work?"):

    """The **SHAP** package takes a given model as an input, then does the following for a given feature `F`:

    1. Create the set of all possible feature combinations (called coalitions)
    2. Calculate the average model prediction
    3. For each coalition, calculate the difference between the model's prediction without `F` and the average prediction.
    4. For each coalition, calculate the difference between the model's prediction with `F` and the average prediction.
    5. For each coalition, calculate how much `F` changed the model's prediction from the average (i.e., step 4 - step 3) - this is the marginal contribution of `F`.
    6. Shapley value = the average of all the values calculated in step 5 (i.e., the average of `F's` marginal contributions)

    """

    "[Source](https://www.h2o.ai/blog/shapley-values-a-gentle-introduction/)"

with st.sidebar.expander("Data Dictionary"):
    """
    **CRIM**: Per capita crime rate by town

    **ZN**: Proportion of residential land zoned for lots over 25,000 sq. ft

    **INDUS**: Proportion of non-retail business acres per town

    **CHAS**: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)

    **NOX**: Nitric oxide concentration (parts per 10 million)

    **RM**: Average number of rooms per dwelling

    **AGE**: Proportion of owner-occupied units built prior to 1940

    **DIS**: Weighted distances to five Boston employment centers

    **RAD**: Index of accessibility to radial highways

    **TAX**: Full-value property tax rate per $10,000

    **PTRATIO**: Pupil-teacher ratio by town

    **B**: 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town

    **LSTAT**: Percentage of lower status of the population

    **MEDV**: Median value of owner-occupied homes in $1000s
    """

## main content area

with st.expander("Model Features and Data summary"):
    nobs = st.slider(
        "Select number of observations to visually inspect", 1, X.shape[0], value=20
    )
    # Display data
    st.dataframe(X.head(nobs))

    # Conditionally calculate summary statistics
    if st.checkbox("Display summary statistics for visible sample?"):
        f"""Sample statistics based on {nobs} observations:"""
        st.dataframe(X.head(nobs).describe())


"## Using the SHAP library"

"To calculate SHAP values, we first pass the model to `shap.TreeExplainer`, then tell the library to calculate the model SHAP values:"
with st.echo():
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

"With the object `shap_values` created, we can now embed various graphics that explain how our housing model works:"

st_shap(shap.force_plot(explainer.expected_value, shap_values, X), height=350)

"""
The interactive graph allows you to select your Y and X values, and dynamically generate plots to understand the model
"""

"---"

"## Organizing outputs into columns"

"Apps can be rendered top-to-bottom, or you can organize output widgets into columns"

c1, c2 = st.columns(2)

with c1:
    "### Feature importance - All"
    st_shap(shap.summary_plot(shap_values, X))


with c2:
    feature = st.selectbox("Choose variable", X.columns)
    f"### {feature} importance on housing price"
    st_shap(shap.dependence_plot(feature, shap_values, X))
