# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://localhost:8000"  # Adjust if deployed elsewhere


def main():
    st.title("Git-Based Experiment Dashboard")

    # Button to fetch experiments from FastAPI
    if st.button("Refresh Experiments"):
        experiments = fetch_experiments()
        if experiments:
            st.session_state["experiments"] = experiments

    # Display experiments
    if "experiments" not in st.session_state:
        st.session_state["experiments"] = fetch_experiments()

    exp_list = st.session_state["experiments"]
    if exp_list:
        df = pd.json_normalize(exp_list)
        st.write("### Experiment List")
        st.dataframe(df)

        # Simple selection of an experiment row
        selected_index = st.selectbox(
            "Select an experiment to inspect", range(len(exp_list))
        )
        selected_exp = exp_list[selected_index]

        st.write("### Experiment Details")
        st.json(selected_exp)

        # Potentially add charts based on `metrics` here
        if "metrics" in selected_exp:
            metrics_df = pd.DataFrame([selected_exp["metrics"]])
            st.bar_chart(metrics_df.T)  # Very basic example
    else:
        st.write(
            "No experiments found. Try logging some experiments via the API."
        )


def fetch_experiments():
    try:
        resp = requests.get(f"{API_BASE_URL}/experiments")
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(
                f"Failed to fetch experiments. Status code: {resp.status_code}"
            )
            return []
    except Exception as e:
        st.error(f"Error fetching experiments: {e}")
        return []


if __name__ == "__main__":
    main()
