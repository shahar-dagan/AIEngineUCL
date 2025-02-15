# streamlit_app.py
import streamlit as st
import pandas as pd
from experiment_manager import ExperimentManager
from pydantic import BaseModel
from typing import Optional


# Create a single instance of ExperimentManager
@st.cache_resource
def get_experiment_manager():
    return ExperimentManager()


class Experiment(BaseModel):
    name: str
    hyperparams: dict
    metrics: dict
    notes: Optional[str] = None


def main():
    st.title("Git-Based Experiment Dashboard")

    # Get ExperimentManager instance
    exp_manager = get_experiment_manager()

    # Create tabs for viewing vs logging
    tab1, tab2 = st.tabs(["View Experiments", "Log New Experiment"])

    with tab1:
        if st.button("Refresh Experiments"):
            st.session_state["experiments"] = exp_manager.load_all_experiments()

        # Initialize experiments in session state if not present
        if "experiments" not in st.session_state:
            st.session_state["experiments"] = exp_manager.load_all_experiments()

        exp_list = st.session_state["experiments"]
        if exp_list:
            df = pd.json_normalize(exp_list)

            # Add filtering options
            st.write("### Filters")
            col1, col2 = st.columns(2)
            with col1:
                min_accuracy = st.slider("Min Accuracy", 0.0, 1.0, 0.0)
            with col2:
                max_loss = st.slider("Max Loss", 0.0, 10.0, 10.0)

            # Apply filters
            filtered_df = df[
                (df["metrics.accuracy"] >= min_accuracy)
                & (df["metrics.loss"] <= max_loss)
            ]

            st.write("### Experiment List")
            st.dataframe(filtered_df)

            # Experiment details
            st.write("### Experiment Details")
            selected_index = st.selectbox(
                "Select an experiment to inspect", range(len(exp_list))
            )
            selected_exp = exp_list[selected_index]
            st.json(selected_exp)

            # Visualizations
            if "metrics" in selected_exp:
                st.write("### Metrics Visualization")
                metrics_df = pd.DataFrame([exp["metrics"] for exp in exp_list])
                metrics_df.index = [exp["name"] for exp in exp_list]
                st.line_chart(metrics_df)

    with tab2:
        st.write("### Log New Experiment")
        exp_name = st.text_input("Experiment Name")

        # Hyperparameters input
        st.write("#### Hyperparameters")
        lr = st.number_input(
            "Learning Rate", min_value=0.0001, max_value=1.0, value=0.001
        )
        batch_size = st.number_input(
            "Batch Size", min_value=1, max_value=512, value=32
        )

        # Metrics input
        st.write("#### Metrics")
        accuracy = st.number_input(
            "Accuracy", min_value=0.0, max_value=1.0, value=0.0
        )
        loss = st.number_input("Loss", min_value=0.0, value=0.0)

        notes = st.text_area("Notes")

        if st.button("Log Experiment"):
            try:
                experiment_data = {
                    "name": exp_name,
                    "hyperparams": {"lr": lr, "batch_size": batch_size},
                    "metrics": {"accuracy": accuracy, "loss": loss},
                    "notes": notes,
                }

                # Validate with Pydantic
                exp = Experiment(**experiment_data)

                # Save experiment
                exp_manager.save_experiment(exp.dict())

                # Update session state
                st.session_state["experiments"] = (
                    exp_manager.load_all_experiments()
                )
                st.success("Experiment logged successfully!")

            except Exception as e:
                st.error(f"Error logging experiment: {str(e)}")


if __name__ == "__main__":
    main()
