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

    # Sort DataFrame by timestamp to ensure line connects points in chronological order
    df = df.sort_values("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d-%H%M%S")

    # Create timeline
    fig = go.Figure()

    # Add points and lines
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["accuracy"],
            mode="lines+markers",  # Changed to include both lines and markers
            marker=dict(size=15),
            line=dict(width=2),  # Added line width
            name="Models",
            hovertemplate=(
                "<b>Model:</b> %{customdata[0]}<br>"
                "<b>Time:</b> %{x}<br>"
                "<b>Accuracy:</b> %{y:.3f}<br>"
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

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Radio buttons for model selection
    st.write("### Select Model for Details")
    selected_model = st.radio(
        "Select a model",
        options=df["name"].tolist(),
        format_func=lambda x: f"{x} (Accuracy: {df[df['name']==x]['accuracy'].iloc[0]:.3f})",
        label_visibility="collapsed",
    )

    # Show details if a model is selected
    if selected_model:
        exp = df[df["name"] == selected_model].iloc[0]
        st.write("---")
        st.subheader(f"Model Details: {exp['name']}")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Performance:**")
            st.write(f"- Accuracy: {exp['accuracy']:.3f}")

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

        with col2:
            st.write("**Hyperparameters:**")
            for key, value in exp["hyperparameters"].items():
                st.write(f"- {key}: {value}")


if __name__ == "__main__":
    main()
