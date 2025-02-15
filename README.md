# Setup

source venv/bin/activate

# Run the FastAPI server

uvicorn app:app --reload --port 8000

# Run the Streamlit app

streamlit run streamlit_app.py --server.port 8501

# Log an experiment

curl -X POST -H "Content-Type: application/json" \
 -d '{
"name": "my_cool_experiment",
"hyperparams": {"lr": 0.001, "batch_size": 32},
"metrics": {"accuracy": 0.89, "loss": 0.4},
"notes": "Test run"
}' \
 http://localhost:8000/experiments
