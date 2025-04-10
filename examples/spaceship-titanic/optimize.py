from pathlib import Path
import pandas as pd


def predict(test: Path, save: Path):
    # TODO: Add a model here

    test_data = pd.read_csv(test)
    submission = pd.DataFrame(
        {"PassengerId": test_data["PassengerId"], "Transported": False}
    )
    submission.to_csv(save, index=False)
    print(f"Test submission saved to {save}")


if __name__ == "__main__":
    # This block is primarily for testing the script directly,
    # it's not used by the weco evaluation loop.
    script_dir = Path(__file__).resolve().parent
    # Use validation data as test data *for this test block only*
    train_file_path = script_dir / "public" / "train.csv"
    print("train_file_path:", train_file_path)
    test_file_path = script_dir / "public" / "test.csv"
    print("test_file_path:", test_file_path)
    submission_output_path = script_dir / "submission.csv"

    # Call predict with the DataFrame and the correct output path
    predict(train_file_path, test_file_path, submission_output_path)
    print(f"Test submission saved to {submission_output_path}")
