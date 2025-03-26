import os
from typing import Any, Dict, Optional
import wandb
from git import Repo


def get_git_sha() -> str:
    repo = Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def init_wandb(
    project: str,
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:

    if config is None:
        config = {}

    config["git_sha"] = get_git_sha()

    wandb.init(
        project=project,
        name=name,
        config=config,
    )


def log_eval_results(
    summary_results: Dict[str, Any],
    model_name: str,
    task_name: str,
) -> None:
    wandb.log(
        {
            "accuracy": summary_results["accuracy"],
            "total_completion_tokens": summary_results["total_completion_tokens"],
            "avg_completion_tokens": summary_results["avg_completion_tokens"],
            "total_prompt_tokens": summary_results["total_prompt_tokens"],
            "avg_prompt_tokens": summary_results["avg_prompt_tokens"],
            "model": model_name,
            "task": task_name,
        }
    )

    # Log pass@k metrics if available
    if summary_results.get("pass_at_k"):
        for k, v in summary_results["pass_at_k"].items():
            wandb.log({f"pass@{k}": v})

    wandb.log(summary_results["configuration"])


def finish_wandb() -> None:
    wandb.finish()
