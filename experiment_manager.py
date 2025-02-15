# experiment_manager.py
import json
import os
from datetime import datetime
import git
from typing import List, Dict

EXPERIMENTS_DIR = "experiments"
REPO_PATH = "."  # path to your local Git repo


def save_experiment(experiment_data: dict):
    """
    Saves experiment data to a JSON file, then commits via Git.
    """
    if not os.path.exists(EXPERIMENTS_DIR):
        os.makedirs(EXPERIMENTS_DIR)

    # Create a filename with a timestamp, e.g. "experiment_2025-02-15_1530.json"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = experiment_data.get("name", "unnamed")
    file_path = os.path.join(EXPERIMENTS_DIR, f"{exp_name}_{timestamp}.json")

    # Write JSON
    with open(file_path, "w") as f:
        json.dump(experiment_data, f, indent=2)

    # Commit to Git
    repo = git.Repo(REPO_PATH)
    repo.index.add([file_path])
    commit_message = f"Add experiment: {exp_name} @ {timestamp}"
    repo.index.commit(commit_message)


def load_all_experiments() -> List[Dict]:
    """
    Loads all experiments from the experiments folder as a list of dicts.
    """
    if not os.path.exists(EXPERIMENTS_DIR):
        return []

    experiments = []
    for filename in os.listdir(EXPERIMENTS_DIR):
        if filename.endswith(".json"):
            file_path = os.path.join(EXPERIMENTS_DIR, filename)
            with open(file_path, "r") as f:
                data = json.load(f)
                experiments.append(data)
    return experiments
