from catboost import CatBoostRegressor
import shap
import pickle

# Example adapted from:
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Catboost%20tutorial.html

# load dataset
X, y = shap.datasets.boston()

# assume feature engineering
# and other fun stuff here...#

# model estimation
model = CatBoostRegressor(iterations=300, learning_rate=0.1, random_seed=123)
model.fit(X, y, verbose=False, plot=False)

# save "final" model as .pkl to give to client
pickle.dump(model, open("model_files/housing_catboost.pkl", "wb"))
