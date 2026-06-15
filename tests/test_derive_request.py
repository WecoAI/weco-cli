from weco.commands.start.tui_bridge import build_derive_prompt, parse_derive_paths


def test_parse_derive_paths_keeps_valid_entries_and_drops_garbage():
    paths = parse_derive_paths(
        {
            "paths": [
                {"node_id": "n1", "step": 4, "instructions": "  go fast  ", "steps": 25},
                {"node_id": "n2", "steps": 0},
                {"node_id": "n3", "steps": True},
                {"node_id": ""},
                {"instructions": "orphan"},
                "not-a-dict",
            ]
        }
    )
    assert paths == [
        {"node_id": "n1", "step": 4, "instructions": "go fast", "steps": 25},
        {"node_id": "n2"},
        {"node_id": "n3"},
    ]


def test_parse_derive_paths_handles_missing_or_malformed_list():
    assert parse_derive_paths({}) == []
    assert parse_derive_paths({"paths": "nope"}) == []


def test_build_derive_prompt_emits_one_command_per_path_with_quoted_instructions():
    prompt = build_derive_prompt("run-1", [{"node_id": "n1", "instructions": 'try "cuda" graphs'}, {"node_id": "n2"}])
    assert "weco run derive run-1 --from-step n1 -i 'try \"cuda\" graphs' --output plain" in prompt
    assert "weco run derive run-1 --from-step n2 --output plain" in prompt
    assert "run_in_background" in prompt
    assert "weco run status" in prompt


def test_build_derive_prompt_passes_per_path_step_count():
    prompt = build_derive_prompt("run-1", [{"node_id": "n1", "steps": 50}, {"node_id": "n2"}])
    assert "weco run derive run-1 --from-step n1 -n 50 --output plain" in prompt
    assert "weco run derive run-1 --from-step n2 --output plain" in prompt
