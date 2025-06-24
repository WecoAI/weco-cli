import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class InvalidSubmissionError(Exception):
    pass


def evaluate_for_accuracy(
    submission_df: pd.DataFrame, answers_df: pd.DataFrame, target_column: str = "Transported", id_column: str = "PassengerId"
) -> float:
    # Answers checks
    assert target_column in answers_df.columns, f"Answers must have a `{target_column}` column"
    assert id_column in answers_df.columns, f"Answers must have a `{id_column}` column"

    # Submission checks
    if len(submission_df) != len(answers_df):
        raise InvalidSubmissionError("Submission must have the same length as the answers.")
    if target_column not in submission_df.columns:
        raise InvalidSubmissionError(f"Submission must have a `{target_column}` column")
    if id_column not in submission_df.columns:
        raise InvalidSubmissionError(f"Submission must have a `{id_column}` column")

    # Sort on id to ensure correct ordering
    submission_df = submission_df.sort_values(by=id_column)
    answers_df = answers_df.sort_values(by=id_column)

    if (submission_df[id_column].values != answers_df[id_column].values).any():
        raise InvalidSubmissionError(f"Submission and Answers `{id_column}`'s do not match")

    return accuracy_score(submission_df[target_column], answers_df[target_column])


def read_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(data_dir / "train.csv")
    train_df, validation_df = train_test_split(train_df, test_size=0.1, random_state=0)
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, validation_df, test_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("./data/"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    train_df, validation_df, test_df = read_data(args.data_dir)

    # Import training and prediction functions
    from train import train_model, predict_with_model

    # Validate that required functions exist and are callable
    assert callable(train_model), "train_model function must exist and be callable"
    assert callable(predict_with_model), "predict_with_model function must exist and be callable"

    # Step 1: Train the model (this will be optimized by Weco)
    print("Training model...")
    model = train_model(train_df, args.seed)

    # Step 2: Generate predictions on validation set (no retraining)
    print("Generating validation predictions...")
    validation_submission_df = predict_with_model(model, validation_df)

    # Step 3: Evaluate accuracy on validation set
    acc = evaluate_for_accuracy(validation_submission_df, validation_df)
    print(f"accuracy: {acc:.6f}")

    # Step 4: Generate predictions on test set (optional, for final submission)
    print("Generating test predictions...")
    test_submission_df = predict_with_model(model, test_df)
    test_submission_df.to_csv("submission.csv", index=False)
    print("Test predictions saved to submission.csv")
