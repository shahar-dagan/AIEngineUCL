import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime

# Constants
MODEL_CACHE_DIR = Path("models")  # Directory where models are saved


def load_experiment_metadata():
    """Load metadata from all experiments in the model cache"""
    experiments = []

    # Check if directory exists
    if not MODEL_CACHE_DIR.exists():
        st.error(f"Directory not found: {MODEL_CACHE_DIR}")
        return experiments

    # Iterate through all model directories
    for model_dir in MODEL_CACHE_DIR.glob("model_*"):
        if model_dir.is_dir():
            metadata_file = model_dir / "model_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        metadata["path"] = str(model_dir)
                        metadata["name"] = model_dir.name
                        experiments.append(metadata)
                except Exception as e:
                    st.error(f"Error loading {metadata_file}: {str(e)}")

    return experiments


def create_timeline(experiments):
    """Create interactive timeline visualization"""
    df = pd.DataFrame(
        [
            {
                "timestamp": exp["timestamp"],
                "name": exp["name"],
                "accuracy": exp["test_data"]["test_accuracy"],
                "loss": exp["train_data"]["train_loss"][-1],
                "hyperparameters": exp["hyperparameters"],
                "path": exp["path"],
            }
            for exp in experiments
        ]
    )

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d-%H%M%S")

    # Create figure
    fig = go.Figure()

    # Add points for each experiment
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=[0] * len(df),  # All points on same horizontal line
            mode="markers",
            marker=dict(size=15),
            name="Experiments",
            hovertemplate=(
                "<b>Experiment:</b> %{customdata[0]}<br>"
                "<b>Time:</b> %{x}<br>"
                "<b>Accuracy:</b> %{customdata[1]:.3f}<br>"
                "<b>Loss:</b> %{customdata[2]:.3f}<br>"
                "<b>Learning Rate:</b> %{customdata[3]}<br>"
                "<b>Batch Size:</b> %{customdata[4]}<br>"
                "<b>Conv Filters:</b> %{customdata[5]}"
            ),
            customdata=list(
                zip(
                    df["name"],
                    df["accuracy"],
                    df["loss"],
                    df["hyperparameters"].apply(lambda x: x["learning_rate"]),
                    df["hyperparameters"].apply(lambda x: x["batch_size"]),
                    df["hyperparameters"].apply(lambda x: x["conv_filters"]),
                )
            ),
        )
    )

    # Update layout
    fig.update_layout(
        title="Model Training Timeline",
        showlegend=False,
        yaxis=dict(
            showticklabels=False, showgrid=False, zeroline=False
        ),  # Hide y-axis
        xaxis_title="Time",
        hovermode="closest",
        height=200,  # Reduced height since we only need one line
    )

    return fig, df


def main():
    st.title("Model Training Timeline")

    # Load experiments
    experiments = load_experiment_metadata()

    if not experiments:
        st.warning("No experiments found.")
        return

    # Create and display timeline
    fig, df = create_timeline(experiments)
    st.plotly_chart(fig, use_container_width=True)

    # When a point is clicked, show detailed information
    selected_point = st.session_state.get("selected_point", None)
    if selected_point is not None:
        exp = df.iloc[selected_point]
        st.write("### Experiment Details")
        st.write(f"**Time:** {exp['timestamp']}")
        st.write(f"**Accuracy:** {exp['accuracy']:.3f}")
        st.write(f"**Loss:** {exp['loss']:.3f}")
        st.write("**Hyperparameters:**")
        for key, value in exp["hyperparameters"].items():
            st.write(f"- {key}: {value}")


if __name__ == "__main__":
    main()
