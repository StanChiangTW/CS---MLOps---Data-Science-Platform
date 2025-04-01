import json
import logging
from fastapi import FastAPI, HTTPException
from dsba.model_registry import list_models_ids, load_model, load_model_metadata, list_datasets
from dsba.model_prediction import classify_record
import os
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S,",
)

app = FastAPI()


# using FastAPI with defaults is very convenient
# we just add this "decorator" with the "route" we want.
# If I deploy this app on "https//mywebsite.com", this function can be called by visiting "https//mywebsite.com/models/"
@app.get("/models/")
async def list_models():
    return list_models_ids()


@app.get("/datasets/")
async def list_available_datasets():
    return list_datasets()



@app.api_route("/predict/", methods=["GET", "POST"])
async def predict(model_id: str, query: str = None, dataset_name: str = None):
    """
    Predict the target column of a record using a model with a dataset (dataset_name) 
    (idea to develop in the future: with a unique record (query))
    """
    try:
        model = load_model(model_id)
        metadata = load_model_metadata(model_id)

        dataset_path = os.path.join("data", dataset_name)

        df = pd.read_csv(dataset_path)

        predictions = model.predict(df.drop(columns=[metadata.target_column]))
        return {"type": "dataset", "predictions": predictions.tolist()}

 
    except Exception as e:
        # We do want users to be able to see the exception message in the response
        # FastAPI will by default block the Exception and send a 500 status code
        # (In the HTTP protocol, a 500 status code just means "Internal Server Error" aka "Something went wrong but we're not going to tell you what")
        # So we raise an HTTPException that contains the same details as the original Exception and FastAPI will send to the client.
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/favicon.ico")
async def favicon():
    return FileResponse("path_to_some_empty_icon.ico")