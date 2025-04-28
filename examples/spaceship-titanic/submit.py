import argparse
import kaggle
from pathlib import Path


def submit_submission(submission_path: Path):
    kaggle.api.competition_submit(
        submission_path, "My first submission with weco", "spaceship-titanic"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-path", "-p", type=Path, default="submission.csv")
    args = parser.parse_args()
    submit_submission(args.submission_path)
