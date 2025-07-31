import pathlib
import shlex
import argparse
from typing import List, Optional, Dict, Any, Tuple
import sys

from rich.console import Console
from rich.prompt import Prompt
from rich.live import Live
from gitingest import ingest

from .api import (
    get_optimization_suggestions_from_codebase,
    generate_evaluation_script_and_metrics,
    analyze_script_execution_requirements,
    analyze_evaluation_environment,
)
from .panels import OptimizationOptionsPanel, EvaluationScriptPanel


class UserInteractionHelper:
    """Helper class for standardized user interactions."""

    def __init__(self, console: Console):
        self.console = console

    def get_choice(
        self, prompt: str, choices: List[str], default: str = None, show_choices: bool = True, max_retries: int = 5
    ) -> str:
        """Standardized choice prompt with error handling."""
        return self._get_choice_with_retry(prompt, choices, default, show_choices, max_retries)

    def _get_choice_with_retry(
        self, prompt: str, choices: List[str], default: str = None, show_choices: bool = True, max_retries: int = 5
    ) -> str:
        """Get user choice with retry logic and error handling."""
        attempts = 0

        while attempts < max_retries:
            try:
                # Use Rich's Prompt.ask which handles basic validation
                response = Prompt.ask(prompt, choices=choices, default=default, show_choices=show_choices)
                return response
            except (KeyboardInterrupt, EOFError):
                # Handle Ctrl+C or Ctrl+D gracefully
                self.console.print("\n[yellow]Operation cancelled by user.[/]")
                raise
            except Exception:
                attempts += 1
                self.console.print("\n[red]Invalid option.[/]")

                if attempts >= max_retries:
                    self.console.print(f"[red]Maximum retry attempts ({max_retries}) reached. Exiting.[/]")
                    raise Exception("Maximum retry attempts exceeded. Please try again.")

                # Show available options without the full prompt
                if choices:
                    if len(choices) <= 10:  # Show all options if not too many
                        options_str = " / ".join([f"[bold]{choice}[/]" for choice in choices])
                        self.console.print(f"Valid options: {options_str}")
                    else:
                        self.console.print(f"Please enter a valid option from the {len(choices)} available choices.")

                if default:
                    self.console.print(f"Press Enter for default: [bold]{default}[/]")

                continue

        # This should never be reached due to the exception above, but just in case
        raise Exception("Unexpected error while selecting a choice")

    def get_choice_numeric(self, prompt: str, max_number: int, default: int = None, max_retries: int = 5) -> int:
        """Get numeric choice with validation and error handling."""
        choices = [str(i + 1) for i in range(max_number)]
        default_str = str(default) if default is not None else None

        attempts = 0
        while attempts < max_retries:
            try:
                response = Prompt.ask(prompt, choices=choices, default=default_str, show_choices=False)
                return int(response)
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Operation cancelled by user.[/]")
                raise
            except (ValueError, Exception):
                attempts += 1
                self.console.print("\n[red]Invalid option.[/]")

                if attempts >= max_retries:
                    self.console.print(f"[red]Maximum retry attempts ({max_retries}) reached. Exiting.[/]")
                    raise Exception("Maximum retry attempts exceeded. Please try again.")

                # Show valid range
                self.console.print(f"Please enter a number between [bold]1[/] and [bold]{max_number}[/]")
                if default_str:
                    self.console.print(f"Press Enter for default: [bold]{default_str}[/]")

                continue

        raise Exception("Unexpected error in numeric choice selection")

    def get_yes_no(self, prompt: str, default: str = "y", max_retries: int = 5) -> bool:
        """Standardized yes/no prompt with error handling."""
        attempts = 0

        while attempts < max_retries:
            try:
                response = Prompt.ask(prompt, choices=["y", "n"], default=default).lower()
                return response in ["y", "yes"]
            except (KeyboardInterrupt, EOFError):
                self.console.print("\n[yellow]Operation cancelled by user.[/]")
                raise
            except Exception:
                attempts += 1
                self.console.print("\n[red]Invalid option.[/]")

                if attempts >= max_retries:
                    self.console.print(f"[red]Maximum retry attempts ({max_retries}) reached. Exiting.[/]")
                    raise Exception("Maximum retry attempts exceeded. Please try again.")

                self.console.print("Valid options: [bold]y[/] / [bold]n[/]")
                if default:
                    self.console.print(f"Press Enter for default: [bold]{default}[/]")

                continue

        raise Exception("Unexpected error while selecting an option")

    def display_optimization_options_table(self, options: List[Dict[str, str]]) -> None:
        """Display optimization options in a formatted table."""
        options_panel = OptimizationOptionsPanel()
        table = options_panel.get_display(options)
        self.console.print(table)

    def display_selection_confirmation(self, item_text: str) -> None:
        """Display a confirmation message for user selection."""
        self.console.print(f"\n[bold blue]Selected:[/] [bold cyan]{item_text}[/]")
        self.console.print()

    def get_multiline_input(self, intro_message: str) -> str:
        """Handle multiline input with proper instructions."""
        self.console.print(intro_message)
        self.console.print("[dim]Current script content will be replaced[/dim]\n")
        edited_lines = []
        try:
            while True:
                line = input()
                edited_lines.append(line + "\n")
        except EOFError:
            pass
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Edit cancelled.[/]")
            return ""
        return "".join(edited_lines)


class Chatbot:
    def __init__(
        self, project_path: pathlib.Path, console: Console, run_parser: argparse.ArgumentParser, model: Optional[str] = None
    ):
        self.project_path = project_path
        self.console = console
        self.ui_helper = UserInteractionHelper(console)
        self.weco_run_parser = run_parser
        self.user_specified_model = model  # Store user's model choice
        self.resolved_model = None  # Will be determined during workflow
        # State tracking (replacing conversation manager)
        self.current_step: str = "eval_analysis"
        self.evaluation_analysis: Optional[Dict[str, Any]] = None
        self.selected_eval_config: Optional[Dict[str, Any]] = None

        # GitIngest data
        self.gitingest_summary: Optional[str] = None
        self.gitingest_tree: Optional[str] = None
        self.gitingest_content: Optional[Dict[str, str]] = None
        self.gitingest_content_str: Optional[str] = None

        # Chat UI components (removed unused chat layout)
        self.active_live_display: Optional[Live] = None
        self.use_chat_ui = False

    def analyze_codebase_and_get_optimization_options(self) -> Optional[List[Dict[str, str]]]:
        """Analyze the codebase using gitingest and get optimization suggestions from Gemini."""
        try:
            with self.console.status("[bold green]Parsing codebase...[/]"):
                result = ingest(
                    str(self.project_path),
                    exclude_patterns=set(
                        [
                            "*.git",
                            "*.gitignore",
                            "LICENSE*",
                            "CONTRIBUTING*",
                            "CODE_OF_CONDUCT*",
                            "CHANGELOG*",
                            "*.repomixignore",
                            "*.dockerignore",
                            "*.pyc",
                            "*.pyo",
                            "*.csv",
                            "*.json",
                            "*.jsonl",
                            "*.txt",
                            "*.md",
                            "*.rst",
                            "*.yml",
                            "*.yaml",
                        ]
                    ),
                )
                self.gitingest_summary, self.gitingest_tree, self.gitingest_content_str = result

            if not self.gitingest_content_str:
                self.console.print("[yellow]Warning: gitingest found no content to analyze.[/]")
                return None

            with self.console.status("[bold green]Generating optimization suggestions...[/]"):
                result = get_optimization_suggestions_from_codebase(
                    console=self.console,
                    gitingest_summary=self.gitingest_summary,
                    gitingest_tree=self.gitingest_tree,
                    gitingest_content_str=self.gitingest_content_str,
                )

                if result and isinstance(result, list):
                    options = result  # Use the dictionaries directly from API
                else:
                    options = None

            if not options or not isinstance(options, list):
                self.console.print("[red]Unable to retrieve valid optimization options from the backend.[/]")
                return None

            if not options:
                self.console.print("[yellow]No optimizations suggested for this codebase.[/]")
                return None

            self.ui_helper.display_optimization_options_table(options)
            return options

        except Exception as e:
            self.console.print(f"[bold red]An error occurred during analysis: {e}[/]")
            import traceback

            traceback.print_exc()
            return None

    def get_user_option_selection(self, options: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Get user's selection from the optimization options."""
        if not options:
            return None

        try:
            choice_num = self.ui_helper.get_choice_numeric(
                "\n[bold]Which optimization would you like to pursue?[/bold] (Enter number)", len(options)
            )
            selected_option = options[choice_num - 1]
            self.ui_helper.display_selection_confirmation(selected_option["description"])
            return selected_option
        except Exception as e:
            self.console.print(f"[red]Error selecting optimization option: {e}[/]")
            return None

    def handle_script_generation_workflow(self, selected_option: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Handle script generation, editing, and custom path workflows.

        This implements a state machine for evaluation script creation with these states:
        1. No script: User can Generate or Provide path
        2. Script exists: User can Use, Edit, Regenerate, or Provide different path

        The workflow continues until user chooses to Use a script or exits.
        """
        eval_script_content = None
        eval_script_path_str = None
        metric_name = None
        goal = None

        # Main workflow loop - continues until user accepts a script
        while True:
            if eval_script_content:
                # State: Script exists - show it and offer actions
                self.console.print("\n[bold]Current evaluation script:[/]")
                script_panel = EvaluationScriptPanel()
                panel = script_panel.get_display(eval_script_content, eval_script_path_str or "evaluate.py")
                self.console.print(panel)

                if metric_name and goal:
                    self.console.print(f"\n[green]Suggested metric:[/] {metric_name} (goal: {goal})")

                action = self.ui_helper.get_choice(
                    "Choose an action: [bold]U[/]se this script / [bold]E[/]dit content / [bold]R[/]egenerate / [bold]P[/]rovide different path",
                    ["u", "U", "e", "E", "r", "R", "p", "P"],
                    default="u",
                ).lower()
            else:
                # State: No script - offer initial options
                action = self.ui_helper.get_choice(
                    "How to proceed? ([bold]G[/]enerate / Provide [bold]P[/]ath)", ["g", "G", "p", "P"], default="g"
                ).lower()

            # Action: Use current script (exit workflow)
            if action == "u" and eval_script_content:
                # Save generated script to file if it hasn't been saved yet
                if not eval_script_path_str:
                    eval_script_path_obj = self.project_path / "evaluate.py"
                    eval_script_path_obj.write_text(eval_script_content)
                    eval_script_path_str = "evaluate.py"
                    self.console.print(f"Generated script saved as [cyan]{eval_script_path_str}[/]")
                break

            # Action: Edit current script
            elif action == "e" and eval_script_content:
                eval_script_content = self.ui_helper.get_multiline_input(
                    "\nPlease paste your edited script below. Press Ctrl+D (Unix) or Ctrl+Z then Enter (Windows) when done:"
                )
                if not eval_script_content:
                    continue  # Edit was cancelled, stay in loop

                # After editing, we need new metric info since script changed
                eval_script_path_str = None  # Clear path since content changed
                metric_name = Prompt.ask("Please specify the metric name this script will print")
                goal = self.ui_helper.get_choice(
                    "Should we maximize or minimize this metric?", choices=["maximize", "minimize"], default="maximize"
                )
                continue  # Show the edited script for review

            # Action: Generate new script (or regenerate)
            elif action == "g" or action == "r":
                with self.console.status("[bold green]Generating evaluation script and determining metrics...[/]"):
                    result = generate_evaluation_script_and_metrics(
                        console=self.console,
                        target_file=selected_option["target_file"],
                        description=selected_option["description"],
                        gitingest_content_str=self.gitingest_content_str,
                    )
                if result and result[0]:
                    eval_script_content, metric_name, goal, reasoning = result
                    if reasoning:
                        self.console.print(f"[dim]Reasoning: {reasoning}[/]")
                else:
                    self.console.print(
                        "[red]Unable to generate an evaluation script. Please try providing a custom script path instead.[/]"
                    )
                    eval_script_content = None
                    metric_name = None
                    goal = None
                eval_script_path_str = None  # Generated content not saved yet
                continue  # Show the generated script for review

            # Action: Provide path to existing script
            elif action == "p":
                user_script_path_str = Prompt.ask("Enter the path to your evaluation script (relative to project root)")
                user_script_path = self.project_path / user_script_path_str
                if user_script_path.is_file():
                    try:
                        eval_script_content = user_script_path.read_text()
                        eval_script_path_str = user_script_path_str
                        self.console.print(f"Using script from [cyan]{eval_script_path_str}[/]")

                        # For user-provided scripts, we need manual metric specification
                        metric_name = Prompt.ask("Please specify the metric name this script will print")
                        goal = self.ui_helper.get_choice(
                            "Should we maximize or minimize this metric?", choices=["maximize", "minimize"], default="maximize"
                        )
                        break  # User provided script is ready to use
                    except Exception as e:
                        self.console.print(f"[red]Error reading script {user_script_path_str}: {e}[/]")
                        eval_script_content = None
                else:
                    self.console.print(f"[red]File not found: {user_script_path}[/]")
                continue  # Stay in loop to try again

        # Validate we have all required components
        if not eval_script_content or not eval_script_path_str or not metric_name or not goal:
            return None

        # Analyze the script to determine the proper execution command
        with self.console.status("[bold green]Analyzing script execution requirements...[/]"):
            eval_command = analyze_script_execution_requirements(
                console=self.console,
                script_content=eval_script_content,
                script_path=eval_script_path_str,
                target_file=selected_option["target_file"],
            )

        return {
            "script_path": eval_script_path_str,
            "script_content": eval_script_content,
            "metric_name": metric_name,
            "goal": goal,
            "eval_command": eval_command or f"python {eval_script_path_str}",
        }

    def get_evaluation_configuration(self, selected_option: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Get or create evaluation script configuration using intelligent conversation-guided approach."""
        with self.console.status("[bold green]Analyzing evaluation environment...[/]"):
            analysis = analyze_evaluation_environment(
                console=self.console,
                target_file=selected_option["target_file"],
                description=selected_option["description"],
                gitingest_summary=self.gitingest_summary,
                gitingest_tree=self.gitingest_tree,
                gitingest_content_str=self.gitingest_content_str,
            )

        if not analysis:
            self.console.print("[yellow]Unable to analyze evaluation environment. Falling back to script generation.[/]")
            return self.handle_script_generation_workflow(selected_option)

        self.evaluation_analysis = analysis
        self.current_step = "script_selection"

        return self.handle_evaluation_decision(selected_option, analysis)

    def handle_evaluation_decision(
        self, selected_option: Dict[str, str], analysis: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Handle user decision based on intelligent evaluation analysis.

        This method implements a recommendation system that:
        1. Shows existing evaluation scripts found in the codebase
        2. Provides AI-generated recommendations (use_existing vs generate_new)
        3. Handles user choice with different flows based on recommendation

        The logic adapts the default choice based on AI recommendation to guide users
        toward the most suitable option while still allowing them to override.
        """
        existing_evals = analysis.get("existing_evaluations", [])
        recommendation = analysis.get("recommendation", "generate_new")
        reasoning = analysis.get("reasoning", "")

        # Display existing evaluation scripts if any were found
        if existing_evals:
            from rich.table import Table
            from rich import box

            table = Table(
                title="Existing Evaluation Scripts", show_lines=True, box=box.ROUNDED, border_style="cyan", padding=(1, 1)
            )
            table.add_column("No.", style="bold white", width=5, header_style="bold white", justify="center")
            table.add_column("Script Path", style="cyan", width=20, header_style="bold white")
            table.add_column("Suitability", style="magenta", width=40, header_style="bold white")
            table.add_column("Metrics", style="yellow", width=20, header_style="bold white")
            table.add_column("Confidence", style="green", width=10, header_style="bold white")

            for i, eval_script in enumerate(existing_evals):
                metrics_str = ", ".join([f"{m['name']} ({m['goal']})" for m in eval_script.get("metrics", [])])
                suitability_str = eval_script.get("suitability", "")
                table.add_row(str(i + 1), eval_script["script_path"], suitability_str, metrics_str, eval_script["confidence"])
            self.console.print(table)
        else:
            self.console.print("\n[yellow]No existing evaluation scripts found.[/]")

        # Show AI recommendation with reasoning
        self.console.print(f"\n💡 [bold green]Recommended:[/] [cyan]{recommendation.replace('_', ' ').title()}[/]")
        self.console.print()
        self.console.print(f"[yellow]🧠 Reasoning:[/] {reasoning}")

        # Decision flow 1: AI recommends using existing script
        if existing_evals and recommendation == "use_existing":
            choices = [str(i + 1) for i in range(len(existing_evals))] + ["g"]
            choice = self.ui_helper.get_choice(
                "\n[bold]Choose an option:[/] (Enter number to use existing script, 'g' to generate new)",
                choices,
                default="1" if existing_evals else "g",  # Default to first existing script
                show_choices=False,
            )

            if choice == "g":
                return self.handle_script_generation_workflow(selected_option)
            else:
                selected_eval = existing_evals[int(choice) - 1]
                return self.handle_existing_evaluation_selection(selected_option, selected_eval)

        # Decision flow 2: Scripts exist but AI recommends generating new
        elif existing_evals:
            choices = [str(i + 1) for i in range(len(existing_evals))] + ["g"]
            choice = self.ui_helper.get_choice(
                "\n[bold]Choose an option:[/] (Enter number to use existing script, 'g' to generate new as recommended)",
                choices,
                default="g",  # Default to generate new (following recommendation)
                show_choices=False,
            )

            if choice == "g":
                return self.handle_script_generation_workflow(selected_option)
            else:
                selected_eval = existing_evals[int(choice) - 1]
                return self.handle_existing_evaluation_selection(selected_option, selected_eval)

        # Decision flow 3: No existing scripts found - must generate new
        else:
            self.console.print("\n[cyan]Proceeding to generate a new evaluation script...[/]")
            return self.handle_script_generation_workflow(selected_option)

    def handle_existing_evaluation_selection(
        self, selected_option: Dict[str, str], selected_eval: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Handle selection of an existing evaluation script."""
        script_path = selected_eval["script_path"]

        try:
            script_file = self.project_path / script_path
            if not script_file.exists():
                self.console.print(f"[red]Error: Script file {script_path} not found.[/]")
                return None

            script_content = script_file.read_text()
            self.console.print(f"\n[green]Using existing script:[/] [cyan]{script_path}[/]")
            self.console.print()

            metrics = selected_eval.get("metrics", [])

            if not metrics:
                self.console.print("[yellow]No metrics detected. Please specify manually.[/]")
                metric_name = Prompt.ask("Please specify the metric name this script will print")
                goal = self.ui_helper.get_choice(
                    "Should we maximize or minimize this metric?", choices=["maximize", "minimize"], default="maximize"
                )
            elif len(metrics) == 1:
                metric_name = metrics[0]["name"]
                goal = metrics[0]["goal"]
                self.console.print(f"[green]Using detected metric:[/] [yellow]{metric_name}[/] (goal: {goal})")
            else:
                self.console.print("[green]Multiple metrics detected:[/]")
                for i, m in enumerate(metrics):
                    self.console.print(f"  {i + 1}. {m['name']} (goal: {m['goal']})")
                try:
                    choice_num = self.ui_helper.get_choice_numeric("Which metric to use?", len(metrics))
                    selected_metric = metrics[choice_num - 1]
                    metric_name = selected_metric["name"]
                    goal = selected_metric["goal"]
                except Exception as e:
                    self.console.print(f"[red]Error selecting metric: {e}[/]")
                    return None

            eval_command = selected_eval.get("run_command", "")
            if not eval_command or eval_command == f"python {script_path}":
                with self.console.status("[bold green]Analyzing script execution requirements...[/]"):
                    eval_command = analyze_script_execution_requirements(
                        console=self.console,
                        script_content=script_content,
                        script_path=script_path,
                        target_file=selected_option["target_file"],
                    )

            self.current_step = "confirmation"
            eval_config = {
                "script_path": script_path,
                "script_content": script_content,
                "metric_name": metric_name,
                "goal": goal,
                "eval_command": eval_command or f"python {script_path}",
            }

            self.selected_eval_config = eval_config
            return eval_config

        except Exception as e:
            self.console.print(f"[red]Error processing script {script_path}: {e}[/]")
            return None

    def confirm_and_finalize_evaluation_config(self, eval_config: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Allow user to modify the evaluation command if needed."""
        self.console.print(f"\n[green]Analyzed evaluation command:[/] [cyan]{eval_config['eval_command']}[/]")

        modify_command = self.ui_helper.get_yes_no("Is this the right command to run the evaluation?", default="y")

        if not modify_command:
            self.console.print(f"\n[dim]Current command:[/] {eval_config['eval_command']}")
            new_command = Prompt.ask("Enter the corrected evaluation command", default=eval_config["eval_command"])
            self.console.print(f"[green]Updated command:[/] {new_command}")
            eval_config = {
                "script_path": eval_config["script_path"],
                "script_content": eval_config["script_content"],
                "metric_name": eval_config["metric_name"],
                "goal": eval_config["goal"],
                "eval_command": new_command,
            }

        return eval_config

    def build_weco_command(
        self, target_file: str, steps: int, eval_config: Dict[str, str], model: str, additional_instructions: str = None
    ) -> str:
        """Build the weco command from the optimization and evaluation configs.

        Constructs a properly quoted shell command that can be executed directly.
        Uses shlex.quote() to handle special characters and spaces in arguments safely.
        """
        command_parts = [
            "weco",
            "run",
            "--source",
            shlex.quote(target_file),  # Quote file paths for shell safety
            "--eval-command",
            shlex.quote(eval_config["eval_command"]),  # Quote complex commands with spaces/args
            "--metric",
            shlex.quote(eval_config["metric_name"]),  # Quote metric names that might have spaces
            "--goal",
            eval_config["goal"],  # Goal is always "maximize" or "minimize" (no quoting needed)
            "--steps",
            str(steps),  # Convert int to string
            "--model",
            shlex.quote(model),  # Always include resolved model
        ]

        # Add optional parameters if they're specified
        if additional_instructions:
            command_parts.extend(["--additional-instructions", shlex.quote(additional_instructions)])

        return " ".join(command_parts)

    def execute_optimization(
        self,
        eval_config: Dict[str, str],
        target_file: str,
        steps: int,
        model: str,
        additional_instructions: str,
        weco_run_cmd: str,
    ) -> None:
        """Execute the optimization with the given configuration.

        This method handles two execution paths:
        1. Direct execution: Run with the provided configuration
        2. User adjustment: Allow user to modify the command before execution

        If user chooses to adjust, we parse their command to validate it and extract
        the new configuration parameters.
        """
        self.console.print("\n[bold green]🚀 Starting optimization...[/]")
        self.console.print(f"[dim]Command: {weco_run_cmd}[/]\n")

        # Give user option to adjust parameters before execution
        adjust_command = self.ui_helper.get_yes_no(
            f"\n[bold]Current command:[/] [dim]{weco_run_cmd}[/]\n"
            "[dim]You can modify the evaluation command, steps, model, or other parameters.[/]\n"
            "Would you like to adjust any parameters?",
            default="no",
        )

        if adjust_command:
            # User wants to modify the command - get their input
            new_weco_run_cmd_str = Prompt.ask("Enter the full weco run command", default=weco_run_cmd)

            # Parse the user's command safely using shlex
            try:
                command_tokens = shlex.split(new_weco_run_cmd_str)
            except ValueError as e:
                self.console.print(f"[bold red]Error parsing command: {e}. Please check quotes.[/]")
                return

            # Validate command structure (must start with "weco run")
            if not command_tokens or command_tokens[0] != "weco" or (len(command_tokens) > 1 and command_tokens[1] != "run"):
                self.console.print("[bold red]Invalid command. Must start with 'weco run'.[/]")
                return

            # Extract arguments for the 'run' subcommand (skip "weco run")
            run_args_list = command_tokens[2:]

            try:
                # Parse the arguments using the same parser that CLI uses
                parsed_ns = self.weco_run_parser.parse_args(run_args_list)

                # Update configuration from parsed arguments
                eval_config = {
                    "script_path": "",  # Not needed for execution
                    "script_content": "",  # Not needed for execution
                    "metric_name": parsed_ns.metric,
                    "goal": parsed_ns.goal,
                    "eval_command": parsed_ns.eval_command,
                }

                target_file = parsed_ns.source
                steps = parsed_ns.steps
                model = parsed_ns.model
                additional_instructions = parsed_ns.additional_instructions

            except Exception as e:
                self.console.print(f"[bold red]Error parsing adjusted command: {e}. Optimization not started.[/]")
                return

        # Import and execute the actual optimization function
        # (Import here to avoid circular imports)
        from .optimizer import execute_optimization as execute_optimization_run

        success = execute_optimization_run(
            source=target_file,
            eval_command=eval_config["eval_command"],
            metric=eval_config["metric_name"],
            goal=eval_config["goal"],
            steps=steps,
            model=model,
            log_dir=".runs",  # Standard log directory
            additional_instructions=additional_instructions,
            console=self.console,
        )

        # Report final result to user
        if success:
            self.console.print("\n[bold green]✅ Optimization completed successfully![/]")
        else:
            self.console.print("\n[bold yellow]⚠️  Optimization ended early or encountered issues.[/]")

        exit_code = 0 if success else 1
        sys.exit(exit_code)

    def show_and_copy_command(self, command: str) -> None:
        """Show the command and copy it to clipboard."""
        import subprocess
        import platform

        try:
            if platform.system() == "Darwin":
                subprocess.run(["pbcopy"], input=command.encode(), check=True)
            elif platform.system() == "Linux":
                subprocess.run(["xclip", "-selection", "clipboard"], input=command.encode(), check=True)
            elif platform.system() == "Windows":
                subprocess.run(["clip"], input=command.encode(), check=True)
            self.console.print("\n[green]✅ Command copied to clipboard! Exiting...[/]")
        except Exception:
            self.console.print("\n[yellow]Could not copy to clipboard automatically.[/]")

    def setup_evaluation(self, selected_option: Dict[str, str]) -> Optional[Tuple[str, Dict[str, str], str, int, str, str]]:
        """Setup evaluation environment for the selected optimization."""
        eval_config = self.get_evaluation_configuration(selected_option)
        if not eval_config:
            self.console.print("[red]Evaluation script setup failed. Please check your script configuration and try again.[/]")
            return None

        eval_config = self.confirm_and_finalize_evaluation_config(eval_config)
        if not eval_config:
            self.console.print("[red]Evaluation configuration failed.[/]")
            return None

        steps = 20
        steps_input = Prompt.ask(f"Number of optimization steps (or press Enter to use {steps})", default=str(steps))
        try:
            steps = int(steps_input)
            steps = max(1, min(1000, steps))
            if steps != int(steps_input):
                self.console.print(f"[yellow]Adjusted to valid range: {steps}[/]")
        except ValueError:
            self.console.print(f"[yellow]Invalid input, using default value: {steps}[/]")

        # Resolve the model to use
        if self.user_specified_model:
            self.resolved_model = self.user_specified_model
        else:
            # Use same default model selection as weco run
            from .utils import determine_default_model, read_api_keys_from_env

            llm_api_keys = read_api_keys_from_env()
            self.resolved_model = determine_default_model(llm_api_keys)

        target_file = selected_option["target_file"]
        additional_instructions = selected_option["description"]

        weco_run_cmd_str = self.build_weco_command(
            target_file, steps, eval_config, self.resolved_model, additional_instructions
        )
        return weco_run_cmd_str, eval_config, target_file, steps, self.resolved_model, additional_instructions

    def start(self):
        self.console.print("[bold cyan]Welcome to Weco![/]")
        self.console.print(f"Let's optimize your codebase in: [cyan]{self.project_path}[/]\n")

        options = self.analyze_codebase_and_get_optimization_options()
        if not options:
            return

        selected_option = self.get_user_option_selection(options)
        if not selected_option:
            return

        result = self.setup_evaluation(selected_option)
        if not result:
            return

        weco_command, eval_config, target_file, steps, model, additional_instructions = result

        self.console.print("\n[bold green]Command:[/]")
        self.console.print(f"[on black white]{weco_command}[/]\n")

        self.console.print(f"[yellow]ℹ️  File paths are relative to: {self.project_path}[/]")

        self.console.print("\n[bold green]🎯 What would you like to do?[/]")
        self.console.print("  [cyan]1.[/] [bold]Run now[/] - Start the optimization immediately")
        self.console.print("  [cyan]2.[/] [bold]Show and copy[/] - Display the command and copy to clipboard")

        execution_choice = self.ui_helper.get_choice(
            "\nEnter your choice", choices=["1", "2"], default="1", show_choices=False
        )

        if execution_choice == "1":
            self.execute_optimization(eval_config, target_file, steps, model, additional_instructions, weco_command)
        else:
            self.show_and_copy_command(weco_command)


def run_onboarding_chatbot(
    project_path: pathlib.Path, console: Console, run_parser: argparse.ArgumentParser, model: Optional[str] = None
):
    try:
        chatbot = Chatbot(project_path, console, run_parser, model)
        chatbot.start()
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred: {e}[/]")
        import traceback

        traceback.print_exc()
