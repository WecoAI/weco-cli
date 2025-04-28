# Example: Optimizing a Kaggle Classification Model (Spaceship Titanic)

This example demonstrates using Weco to optimize a Python script designed for the [Spaceship Titanic Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic/overview). The goal is to improve the model's `accuracy` metric by directly optimizing the evaluate.py

## Setup

1.  Ensure you are in the `examples/spaceship-titanic` directory.
2.  **Kaggle Credentials:** You need your Kaggle API credentials (`kaggle.json`) configured to download the competition dataset. Place the `kaggle.json` file in `~/.kaggle/` or set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables. See [Kaggle API documentation](https://github.com/Kaggle/kaggle-api#api-credentials) for details.
3.  **Install Dependencies:** Install the required Python packages:
    ```bash
    pip install -r requirements-test.txt
    ```
4.  **Prepare Data:** Run the utility script once to download the dataset from Kaggle and place it in the expected `./data/` subdirectories:
    ```bash
    python get_data.py
    ```
    After running `get_data.py`, your directory structure should look like this:
    ```
    .
    ├── competition_description.md
    ├── data
    │   ├── sample_submission.csv
    │   ├── test.csv
    │   └── train.csv
    ├── evaluate.py
    ├── get_data.py
    ├── README.md # This file
    ├── requirements-test.txt
    └── submit.py
    ```

## Optimization Command

Run the following command to start optimizing the model:

```bash
weco run --source evaluate.py \
         --eval-command "python evaluate.py --data-dir ./data" \
         --metric accuracy \
         --maximize true \
         --steps 10 \
         --model gemini-2.5-pro-exp-03-25 \
         --additional-instructions "Improve feature engineering, model choice and hyper-parameters."
         --log-dir .runs/spaceship-titanic
```

## Submit the solution

Once the optimization finished, you can submit your predictions to kaggle to see the results. Make sure `submission.csv` is present and then simply run the following command.

```bash
python submit.py
```

### Explanation

*   `--source evaluate.py`: The script provides a baseline as root node and directly optimize the evaluate.py
*   `--eval-command "python evaluate.py --data-dir ./data/"`: The weco agent will run the `evaluate.py` and update it.
    *   [optional] `--data-dir`: path to the train and test data.
    *   [optional] `--seed`: Seed for reproduce the experiment.
*   `--metric accuracy`: The target metric Weco should optimize.
*   `--maximize true`: Weco aims to increase the accuracy.
*   `--steps 10`: The number of optimization iterations.
*   `--model gemini-2.5-pro-exp-03-25`: The LLM driving the optimization.
*   `--additional-instructions "Improve feature engineering, model choice and hyper-parameters."`: A simple instruction for model improvement or you can put the path to [`comptition_description.md`](./competition_description.md) within the repo to feed the agent more detailed information.

Weco will iteratively modify the feature engineering or modeling code within `evaluate.py`, run the evaluation pipeline, and use the resulting `accuracy` to guide further improvements.