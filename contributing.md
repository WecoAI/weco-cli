# Contributing

We welcome your contributions! To get started:

1.  **Fork & Clone the Repository:**
    ```bash
    git clone https://github.com/WecoAI/weco-cli.git
    cd weco-cli
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -e ".[dev]"
    ```

3.  **Create a Feature Branch:**
    ```bash
    git checkout -b feature/your-feature-name
    ```

4.  **Make Changes:** Ensure your code adheres to our style guidelines and includes relevant tests.

5.  **Lint & Test:** Run these before opening a pull request; CI runs the same checks.
    ```bash
    ruff format . && ruff check .
    pytest tests
    ```

6.  **Commit, Push & Open a PR**: Commit your changes, and open a pull request with a clear description of your enhancements.
