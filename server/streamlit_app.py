# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from experiment_manager import ExperimentManager
from pydantic import BaseModel
from typing import Optional
import git
from datetime import datetime
import pytz


# Create a single instance of ExperimentManager
@st.cache_resource
def get_experiment_manager():
    return ExperimentManager()


class Experiment(BaseModel):
    name: str
    hyperparams: dict
    metrics: dict
    notes: Optional[str] = None


def get_commit_history():
    """Get Git commit history for experiments"""
    exp_manager = get_experiment_manager()
    repo = exp_manager.repo

    commits = []
    for commit in repo.iter_commits():
        # Only include commits that modified files in the experiments directory
        if any(
            "experiments/" in item.a_path
            for item in commit.diff(
                commit.parents[0] if commit.parents else git.NULL_TREE
            )
        ):
            commits.append(
                {
                    "hash": commit.hexsha,
                    "message": commit.message.strip(),
                    "author": commit.author.name,
                    "date": datetime.fromtimestamp(
                        commit.committed_date, tz=pytz.UTC
                    ),
                    "files_changed": [
                        item.a_path
                        for item in commit.diff(
                            commit.parents[0]
                            if commit.parents
                            else git.NULL_TREE
                        )
                    ],
                }
            )
    return commits


def create_interactive_timeline(commits):
    """Create an interactive timeline visualization"""
    # Create DataFrame for timeline
    df = pd.DataFrame(commits)

    # Create the timeline figure
    fig = go.Figure()

    # Add dots for each commit
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=[0] * len(df),
            mode="markers+text",
            name="Commits",
            text=df["message"],
            textposition="top center",
            hovertemplate=(
                "<b>Date:</b> %{x}<br>"
                "<b>Message:</b> %{text}<br>"
                "<b>Author:</b> %{customdata[0]}<br>"
                "<b>Hash:</b> %{customdata[1]}"
            ),
            customdata=list(
                zip(df["author"], df["hash"].apply(lambda x: x[:8]))
            ),
            marker=dict(size=15, symbol="circle", color="blue"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Experiment Timeline",
        showlegend=False,
        yaxis=dict(visible=False),
        xaxis=dict(title="Date"),
        height=300,
        hovermode="closest",
    )

    return fig, df


def main():
    st.title("Git-Based Experiment Dashboard")

    # Get ExperimentManager instance
    exp_manager = get_experiment_manager()

    # Get commit history
    commits = get_commit_history()

    # Create interactive timeline
    fig, commits_df = create_interactive_timeline(commits)

    # Display the timeline
    st.plotly_chart(fig, use_container_width=True)

    # Show selected commit details
    if len(commits) > 0:
        st.write("### Commit Details")

        # Get selected date from timeline click or use latest commit
        selected_point = st.session_state.get("selected_point", None)
        if selected_point is not None:
            selected_commit = commits_df.iloc[selected_point]
        else:
            selected_commit = commits_df.iloc[0]

        # Display commit info
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write(
                f"**Date:** {selected_commit['date'].strftime('%Y-%m-%d %H:%M:%S')}"
            )
            st.write(f"**Message:** {selected_commit['message']}")
            st.write(f"**Author:** {selected_commit['author']}")
            st.write(f"**Commit:** {selected_commit['hash'][:8]}")
            st.write("**Files Changed:**")
            for file in selected_commit["files_changed"]:
                if file.startswith("experiments/"):
                    st.write(f"- {file}")

        with col2:
            if st.button(
                f"Load Experiments at {selected_commit['hash'][:8]}",
                key=selected_commit["hash"],
            ):
                try:
                    # Checkout this commit temporarily
                    current = exp_manager.repo.head.commit
                    exp_manager.repo.git.checkout(selected_commit["hash"])

                    # Load experiments from this commit
                    experiments = exp_manager.load_all_experiments()

                    # Store in session state
                    st.session_state["experiments"] = experiments
                    st.session_state["viewing_commit"] = selected_commit["hash"]

                    # Return to current commit
                    exp_manager.repo.git.checkout(current)

                    st.success(
                        f"Loaded experiments from commit {selected_commit['hash'][:8]}"
                    )
                except Exception as e:
                    st.error(f"Error loading experiments: {str(e)}")


if __name__ == "__main__":
    main()
