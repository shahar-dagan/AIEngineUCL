import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import json
import zipfile
import io
from openai import OpenAI

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
    """Create a zip file containing model files"""
    folder_path = Path(folder_path)
    output_path = folder_path / f"{folder_path.name}.zip"

    try:
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            files_to_include = [
                "model_metadata.json",
                "parameters.h5",
                "train.py",
            ]

            for filename in files_to_include:
                file_path = folder_path / filename
                if file_path.exists():
                    # Add file to zip with its relative path
                    zf.write(file_path, filename)

        # Read the created zip file
        with open(output_path, "rb") as f:
            return f.read()

    finally:
        # Clean up the zip file
        if output_path.exists():
            output_path.unlink()


def get_model_files(folder_path):
    """Get model files for download"""
    folder_path = Path(folder_path)
    files_to_include = ["model_metadata.json", "parameters.h5", "train.py"]

    # Create a zip file in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for filename in files_to_include:
            file_path = folder_path / filename
            if file_path.exists():
                # Read file in binary mode
                with open(file_path, "rb") as f:
                    zip_file.writestr(filename, f.read())

    zip_buffer.seek(0)
    return zip_buffer.read()


def get_historical_performance(experiments):
    """Format historical data for AI analysis"""
    history = []
    for exp in experiments:
        history.append(
            {
                "timestamp": exp["timestamp"],
                "accuracy": exp["test_data"]["test_accuracy"],
                "hyperparameters": exp["hyperparameters"],
            }
        )
    return history


def get_ai_suggestions(history):
    """Get AI suggestions based on historical performance"""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    prompt = f"""
    You are an ML hyperparameter optimization expert. Based on these historical results:
    {json.dumps(history, indent=2)}

    Predict the optimal hyperparameter values that would achieve the best accuracy.
    Consider the trends in historical performance, but predict NEW optimized values.
    
    Respond with ONLY a JSON object in this structure:
    {{
        "learning_rate": (predict optimal value),
        "batch_size": (predict optimal value),
        "epochs": (predict optimal value),
        "kernel_size": (predict optimal value as [x, x]),
        "conv_filters": (predict optimal value),
        "dense_layer_neurons": (predict optimal value),
        "activation": (predict optimal value)
    }}

    Use numerical values except for activation which should be a string.
    Do not include any explanation, only the JSON object.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a predictive ML optimization expert. Output only valid JSON with predicted optimal values.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    try:
        # Extract just the JSON part from the response
        content = response.choices[0].message.content.strip()
        # Remove any markdown code block markers if present
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        st.error(f"Error parsing AI suggestions: {str(e)}")
        return None


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
                "message": exp["message"],
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
            mode="lines+markers",
            marker=dict(size=15),
            line=dict(width=2),
            name="Models",
            hovertemplate=(
                "<b>Model:</b> %{customdata[0]}<br>"
                "<b>Time:</b> %{x}<br>"
                "<b>Accuracy:</b> %{y:.3f}<br>"
                "<b>Message:</b> %{customdata[1]}<br>"
            ),
            customdata=list(zip(df["name"], df["message"])),
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

        # Add message display
        st.write(f"**Message:** {exp['message']}")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Performance:**")
            st.write(f"- Accuracy: {exp['accuracy']:.3f}")

            exp_path = Path(exp["path"])
            if exp_path.exists():
                try:
                    model_data = get_model_files(exp_path)
                    st.download_button(
                        label="📥 Download Model Files",
                        data=model_data,
                        file_name=f"{exp['name']}.zip",
                        mime="application/x-zip-compressed",
                    )
                except Exception as e:
                    st.error(f"Error preparing files: {str(e)}")

        with col2:
            st.write("**Current Hyperparameters:**")
            for key, value in exp["hyperparameters"].items():
                st.write(f"- {key}: {value}")

        # Add AI Analysis section
        st.write("---")
        st.subheader("Suggested Hyperparameters")

        try:
            with st.spinner("Analyzing historical performance..."):
                history = get_historical_performance(experiments)
                suggested_params = get_ai_suggestions(history)
                if suggested_params:
                    # Display suggested parameters
                    st.json(suggested_params)

                    # Save to temporary file
                    temp_file = Path("/tmp/suggested_hyperparameters.json")
                    with open(temp_file, "w") as f:
                        json.dump(suggested_params, f, indent=2)

                    # Read file for download
                    with open(temp_file, "rb") as f:
                        st.download_button(
                            label="💾 Save Suggested Hyperparameters",
                            data=f.read(),
                            file_name="suggested_hyperparameters.json",
                            mime="application/json",
                        )

                    # Clean up
                    temp_file.unlink()
        except Exception as e:
            st.error(f"Error generating AI suggestions: {str(e)}")


if __name__ == "__main__":
    main()
