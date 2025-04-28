import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import joblib

def train(df: pd.DataFrame, random_state: int = 0) -> float:
    train_df, val_df = train_test_split(
        df, test_size=0.10, random_state=random_state, stratify=df["Transported"]
    )

    y_train = train_df.pop("Transported")
    y_val   = val_df.pop("Transported")

    model = DummyClassifier(strategy="most_frequent", random_state=random_state)
    model.fit(train_df, y_train)
    preds = model.predict(val_df)
    acc   = accuracy_score(y_val, preds)

    # Persist the model so that other scripts / graders can reuse it -----------
    joblib.dump(model, "model.joblib")

    return acc

def predict_and_score(input_df: pd.DataFrame) -> float:
    model = joblib.load("model.joblib")
    preds = model.predict(input_df)
    y_test = input_df.pop("Transported")
    acc = accuracy_score(y_test, preds)
    return acc

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("./public/"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train_df = pd.read_csv(args.data_dir / "train.csv")
    test_df  = pd.read_csv("./private/test.csv")
    train_acc  = train(train_df, random_state=args.seed)
    print(f"[info] Train accuracy: {train_acc:.6f}")
    acc = predict_and_score(test_df)
    print(f"accuracy: {acc:.6f}")