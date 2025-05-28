# Example: Solving a Kaggle Competition (Spaceship Titanic)

This example demonstrates using Weco to optimize a Python script designed for the [Spaceship Titanic Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic/overview). The goal is to improve the model's `accuracy` metric by directly optimizing the evaluate.py

## Setup

1.  Ensure you are in the `examples/spaceship-titanic` directory.
2.  `pip install weco`
3.  Set up LLM API Key, `export OPENAI_API_KEY="your_key_here"`
4.  **Install Dependencies:** Install the required Python packages:
    ```bash
    pip install -r requirements-test.txt
    ```

## Optimization Command

Run the following command to start optimizing the model:

```bash
weco run --source train.py \
         --eval-command "python evaluate.py --data-dir ./data --seed 0" \
         --metric accuracy \
         --goal maximize \
         --steps 10 \
         --model o4-mini
         --additional-instructions "Improve feature engineering, model choice and hyper-parameters."
```

### Explanation

*   `--source train.py`: The script provides a baseline as root node and directly optimize the train.py
*   `--eval-command "python evaluate.py --data-dir ./data/"`: The weco agent will run the `evaluate.py`.
    *   [optional] `--data-dir`: path to the train and test data.
    *   [optional] `--seed`: Seed for reproduce the experiment.
*   `--metric accuracy`: The target metric Weco should optimize.
*   `--goal maximize`: Weco aims to increase the accuracy.
*   `--steps 10`: The number of optimization iterations.
*   `--model o4-mini`: The LLM driving the optimization.
*   `--additional-instructions "Improve feature engineering, model choice and hyper-parameters."`: A simple instruction for model improvement or you can put the path to [`comptition_description.md`](./competition_description.md) within the repo to feed the agent more detailed information.

Weco will iteratively update the feature engineering or modeling code within `train.py` guided by the evaluation method defined in `evaluate.py`
