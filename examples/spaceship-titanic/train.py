# Best solution from Weco with a score of 0.8103

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

def train_model(train_df: pd.DataFrame, random_state: int = 0):
    df = train_df.copy()
    # Feature Engineering
    if "Cabin" in df.columns:
        df["Deck"] = df["Cabin"].fillna("Unknown").str[0]
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.").fillna("Unknown")
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    y = df["Transported"]
    X = df.drop(columns=["Transported", "PassengerId", "Cabin", "Name"], errors="ignore")
    # Identify columns
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    # Ensure categorical uniformity
    for col in categorical_cols:
        X[col] = X[col].astype(str)
    # Pipelines
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])
    model = Pipeline([
        ("preproc", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            random_state=random_state
        ))
    ])
    model.fit(X, y)
    return model

def predict_with_model(model, test_df: pd.DataFrame) -> pd.DataFrame:
    df = test_df.copy()
    if "Cabin" in df.columns:
        df["Deck"] = df["Cabin"].fillna("Unknown").str[0]
    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.").fillna("Unknown")
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    passenger_ids = df["PassengerId"].copy()
    X = df.drop(columns=["Transported", "PassengerId", "Cabin", "Name"], errors="ignore")
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categorical_cols:
        X[col] = X[col].astype(str)
    preds = model.predict(X)
    return pd.DataFrame({"PassengerId": passenger_ids, "Transported": preds.astype(bool)})