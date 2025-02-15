import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
import zipfile
import io

# Constants
MODEL_CACHE_DIR = Path("models")


def load_experiment_metadata():
    experiments = []
    if not MODEL_CACHE_DIR.exists():
        st.error(f"Directory not found: {MODEL_CACHE_DIR}")
        return experiments

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


def create_zip_from_folder(folder_path):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for fpath in Path(folder_path).rglob("*"):
            if fpath.is_file():
                relative_path = fpath.relative_to(folder_path)
                zip_file.write(fpath, relative_path)
    zip_buffer.seek(0)
    return zip_buffer


def main():
    st.title("Model Training Timeline")

    experiments = load_experiment_metadata()
    if not experiments:
        st.warning("No experiments found.")
        return

    # Create DataFrame for timeline
    df = pd.DataFrame(
        [
            {
                "timestamp": exp["timestamp"],
                "name": exp["name"],
                "accuracy": exp["test_data"]["test_accuracy"],
                "hyperparameters": exp["hyperparameters"],
                "path": exp["path"],
            }
            for exp in experiments
        ]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d-%H%M%S")

    # Create timeline
    fig = go.Figure()

    # Add points for each model
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["accuracy"],
            mode="markers",
            marker=dict(size=15),
            name="Models",
            hovertemplate=(
                "<b>Model:</b> %{customdata[0]}<br>"
                "<b>Time:</b> %{x}<br>"
                "<b>Accuracy:</b> %{y:.3f}<br>"
                "<b>Click for details</b>"
            ),
            customdata=list(zip(df["name"])),
        )
    )

    fig.update_layout(
        title="Model Timeline",
        xaxis_title="Training Time",
        yaxis_title="Test Accuracy",
        hovermode="closest",
    )

    # Display timeline and handle clicks
    clicked = st.plotly_chart(fig, use_container_width=True)

    # Add model selection dropdown
    selected_model = st.selectbox(
        "Select a model to view details",
        options=df["name"].tolist(),
        format_func=lambda x: f"{x} (Accuracy: {df[df['name']==x]['accuracy'].iloc[0]:.3f})",
    )

    if selected_model:
        exp = df[df["name"] == selected_model].iloc[0]

        # Show model details
        st.subheader(f"Model: {exp['name']}")
        st.write(f"**Accuracy:** {exp['accuracy']:.3f}")
        st.write("**Hyperparameters:**")
        for key, value in exp["hyperparameters"].items():
            st.write(f"- {key}: {value}")

        # Add download button
        exp_path = Path(exp["path"])
        if exp_path.exists():
            zip_buffer = create_zip_from_folder(exp_path)
            st.download_button(
                label="📥 Download Model Files",
                data=zip_buffer,
                file_name=f"{exp['name']}.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
