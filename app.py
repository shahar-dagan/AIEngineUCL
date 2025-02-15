# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import experiment_manager

app = FastAPI()


# Pydantic schema for an experiment
class Experiment(BaseModel):
    name: str
    hyperparams: dict
    metrics: dict
    notes: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}


@app.post("/experiments")
def log_experiment(exp: Experiment):
    """
    Save an experiment record to disk (and Git).
    """
    experiment_manager.save_experiment(exp.dict())
    return {"status": "success", "experiment": exp}


@app.get("/experiments")
def list_experiments():
    """
    Return all tracked experiments.
    """
    experiments = experiment_manager.load_all_experiments()
    return experiments
