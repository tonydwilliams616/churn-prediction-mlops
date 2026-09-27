"""
SageMaker inference script - runs INSIDE the endpoint's container every time a
prediction request comes in.

The SageMaker scikit-learn container looks for four specific function names in
this file and calls them in this order for every request:
    model_fn    -> called ONCE when the container starts, to load the model
    input_fn    -> called per request, to parse the incoming request body
    predict_fn  -> called per request, to run the actual prediction
    output_fn   -> called per request, to format the response

This is the same "script mode" idea as sagemaker_job/entry_point.py during
training - the container provides the infrastructure, your script provides
the four specific behaviors it doesn't know how to do on its own.
"""

import json
import os

import joblib
import pandas as pd


def model_fn(model_dir):
    """
    Called once when the endpoint starts. Loads the trained model and the
    expected feature column order (both saved into the model.tar.gz during
    training by sagemaker_job/entry_point.py).
    """
    model = joblib.load(os.path.join(model_dir, "churn_model.joblib"))
    feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))
    return {"model": model, "feature_columns": feature_columns}


def input_fn(request_body, request_content_type):
    """
    Parses the incoming request. Expects a JSON body - either a single object
    representing one customer's features, or a list of such objects for
    multiple customers in one request.

    Example single-customer body:
        {"tenure": 12, "MonthlyCharges": 70.35, "Contract_One year": true, ...}
    """
    if request_content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_artifacts):
    """
    Runs the actual prediction. Reindexes the incoming data to match the
    exact column order the model was trained on - the same safeguard from
    notebook 08, now enforced automatically on every real request rather
    than something we do by hand.

    fill_value=0 handles any expected column the caller didn't provide
    (e.g. a one-hot column for a category that wasn't relevant to this
    customer) by treating it as 0/False, which is the correct default for
    one-hot encoded columns.
    """
    model = model_artifacts["model"]
    feature_columns = model_artifacts["feature_columns"]

    X = input_data.reindex(columns=feature_columns, fill_value=0)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "predictions": predictions.tolist(),
        "churn_probability": probabilities.tolist(),
    }


def output_fn(prediction, accept):
    """Formats the response as JSON."""
    return json.dumps(prediction), accept
