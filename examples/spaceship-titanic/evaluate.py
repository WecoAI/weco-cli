import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # keep Weco's panel tidy


def train(df: pd.DataFrame, test_df: pd.DataFrame, random_state: int = 0) -> float:
    train_df, val_df = train_test_split(
        df, test_size=0.10, random_state=random_state, stratify=df["Transported"]
    )

    y_train = train_df.pop("Transported")
    y_val = val_df.pop("Transported")

    model = DummyClassifier(strategy="most_frequent", random_state=random_state)
    model.fit(train_df, y_train)
    preds = model.predict(val_df)
    acc = accuracy_score(y_val, preds)

    # **Important**: Keep this step!!!
    # Save the model and generate a submission file on test
    joblib.dump(model, "model.joblib")
    test_preds = model.predict(test_df)
    submission_df = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Transported": test_preds.astype(bool)}
    )
    submission_df.to_csv("submission.csv", index=False)

    return acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("./data/"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train_df = pd.read_csv(args.data_dir / "train.csv")
    test_df = pd.read_csv(args.data_dir / "test.csv")
    acc = train(train_df, test_df, random_state=args.seed)
    print(f"accuracy: {acc:.6f}")
