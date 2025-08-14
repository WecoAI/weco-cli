# Model Development (Kaggle Spaceship Titanic)

This example demonstrates using Weco to optimize a Python script designed for the [Spaceship Titanic Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic/overview).
The goal is to improve the model's `accuracy` metric by optimizing the `train.py`

## Setup

1.  Ensure you are in the `examples/spaceship-titanic` directory.
2.  Install Dependencies:
    ```bash
    pip install weco>=0.2.18 pandas numpy scikit-learn torch xgboost lightgbm catboost
    ```
3. Create your OpenAI API key [here](https://platform.openai.com/api-keys), then run:
    ```bash
    export OPENAI_API_KEY="your_key_here"
    ```

## Run Weco

Run the following command to start optimizing the model:

```bash
weco run --source train.py \
     --eval-command "python evaluate.py --data-dir ./data --seed 0" \
     --metric accuracy \
     --goal maximize \
     --steps 10 \
     --model o4-mini \
     --additional-instructions "Improve feature engineering, model choice and hyper-parameters." \
     --log-dir .runs/spaceship-titanic
```

### Explanation

*   `--source train.py`: The script provides a baseline as root node and directly optimize the train.py.
*   `--eval-command "python evaluate.py --data-dir ./data/ --seed 0"`: The weco agent will run the `evaluate.py`.
    *   [optional] `--data-dir`: path to the train and test data.
    *   [optional] `--seed`: Seed for reproduce the experiment.
*   `--metric accuracy`: The target metric Weco should optimize.
*   `--goal maximize`: Weco aims to increase the accuracy.
*   `--steps 10`: The number of optimization iterations.
*   `--model o4-mini`: The LLM driving the optimization.
*   `--additional-instructions "Improve feature engineering, model choice and hyper-parameters."`: A simple instruction for model improvement or you can put the path to [`comptition_description.md`](./competition_description.md) within the repo to feed the agent more detailed information.
*   `--log-dir .runs/spaceship-titanic`: Specifies the directory where Weco should save logs and results for this run.

Weco will iteratively update the feature engineering or modeling code within `train.py` guided by the evaluation method defined in `evaluate.py`

## Next Steps

With model development covered, you might be curious to see how you can make your AI code run faster, saving you time and more importantly GPU credits. Check out our example on automating kernel engineering in [CUDA](/examples/cuda/README.md) and [Triton](/examples/triton/README.md), or dive into the [CLI Reference](https://docs.weco.ai/cli/cli-reference).
