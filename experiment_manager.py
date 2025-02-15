# experiment_manager.py
import json
import os
from datetime import datetime
import git
from typing import List, Dict


class ExperimentManager:
    def __init__(self, repo_path=None):
        if repo_path is None:
            # Default to the directory containing this file
            self.repo_path = os.path.dirname(os.path.abspath(__file__))
        else:
            self.repo_path = repo_path

        # Initialize experiment directory
        self.experiment_dir = "experiments"
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Initialize Git repo connection
        try:
            self.repo = git.Repo(self.repo_path)
        except git.exc.InvalidGitRepositoryError:
            raise ValueError(
                f"No valid git repository found at {self.repo_path}"
            )

    def save_experiment(self, experiment_data: dict):
        """
        Saves experiment data to a JSON file, then commits via Git.
        """
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        # Create a filename with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = experiment_data.get("name", "unnamed")
        file_path = os.path.join(
            self.experiment_dir, f"{exp_name}_{timestamp}.json"
        )

        # Write JSON
        with open(file_path, "w") as f:
            json.dump(experiment_data, f, indent=2)

        # Commit to Git
        self.repo.index.add([file_path])
        commit_message = f"Add experiment: {exp_name} @ {timestamp}"
        self.repo.index.commit(commit_message)

    def load_all_experiments(self) -> List[Dict]:
        """
        Loads all experiments from the experiments folder as a list of dicts.
        """
        if not os.path.exists(self.experiment_dir):
            return []

        experiments = []
        for filename in os.listdir(self.experiment_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.experiment_dir, filename)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    experiments.append(data)
        return experiments
