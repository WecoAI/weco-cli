"""``weco run instruct <run-id> <instructions>``"""

import json
import sys

from rich.console import Console

from .. import make_client
from ...utils import read_additional_instructions


def handle(run_id: str, instructions: str, console: Console) -> None:
    """Update additional instructions for an active run."""
    client = make_client(console)

    result = client.update_instructions(run_id, read_additional_instructions(instructions))

    if result is None:
        print(json.dumps({"error": "Failed to update instructions. Is the run still active?"}))
        sys.exit(1)

    output = {"run_id": run_id, "additional_instructions": result.get("additional_instructions")}

    print(json.dumps(output, indent=2))
