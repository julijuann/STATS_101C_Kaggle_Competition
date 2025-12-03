import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# load data
train = pd.read_csv("aluminum_coldRoll_train.csv")
test = pd.read_csv("aluminum_coldRoll_testNoY.csv")

target_col = "y_passXtremeDurability"
id_col = "ID"

X = train.drop(columns=[target_col])
y = train[target_col]
X_test = test.copy()

# define categorical and numerical columns
cat_cols = [
    "alloy",
    "cutTemp",
    "rollTemp",
    "topEdgeMicroChipping",
    "blockSource",
    "machineRestart",
    "contourDefNdx",
]

num_cols = [c for c in X.columns if c not in cat_cols + [id_col]]

# build preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# define base xgboost model
xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=101,
)

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("xgb", xgb_base),
])

# hyperparameter grid for cross-validation
param_grid = {
    "xgb__n_estimators": [200, 400],
    "xgb__max_depth": [3, 5],
    "xgb__learning_rate": [0.05, 0.1],
    "xgb__subsample": [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0],
}

# perform 5-fold cross-validation
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_log_loss",
    cv=5,
    verbose=1,
    n_jobs=-1,
)

print("running 5-fold cross-validation...")
grid.fit(X, y)

print("best parameters:", grid.best_params_)
print("best cv neg_log_loss:", grid.best_score_)
print("cv log_loss:", -grid.best_score_)

# train final model using entire dataset
final_model = grid.best_estimator_
final_model.fit(X, y)

# generate test predictions
test_pred = final_model.predict_proba(X_test)[:, 1]
test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

submission = pd.DataFrame({
    "ID": X_test[id_col],
    "y_passXtremeDurability": test_pred,
})

submission.to_csv("team16_xgb_cv_tuned.csv", index=False)
print("saved submission as team16_xgb_cv_tuned.csv")
print(submission.head())
