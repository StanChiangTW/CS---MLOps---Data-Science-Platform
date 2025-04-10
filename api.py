from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import pickle
from src.dsba.data_ingestion.data_ingestion import load_csv
from src.dsba.preprocessing_a import preprocess_data
from src.dsba.model_evaluation import evaluate_models
import io
from pathlib import Path
from typing import Optional
import sys
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import matplotlib.pyplot as plt
from fastapi.templating import Jinja2Templates
import numpy
sys.modules['numpy._core'] = numpy.core





df = load_csv("data/BankChurners.csv")
X_train, X_test, y_train, y_test = preprocess_data("data/BankChurners.csv")

lgbm_model = joblib.load('models/lgbm_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
svm_model = joblib.load('models/svm_model.pkl')

models = {
    "LGBM": lgbm_model,
    "RandomForest": rf_model,
    "XGBoost": xgb_model,
    "SVM": svm_model
}

results, model_comparison = evaluate_models(models, X_train, y_train, X_test, y_test)

png_path = "static/model_comparison.png"
model_comparison.savefig(png_path)



data_info =  df.iloc[:5].to_html(classes='table table-striped table-bordered', index=False)
columns_df = pd.DataFrame({'Column Name': df.columns.tolist()})
columns_info = columns_df.to_html(classes='table table-striped table-bordered', index=False)
model_info = {"Show the model info": {}}
for model_name, model in models.items():
    model_info["Show the model info"][model_name] = model.__class__.__name__

results_dict = results.to_dict(orient="records")
metrics_summary = {}

for record in results_dict:
    model_name = record["Model"]
    dataset_type = record["Dataset"]
    
    if model_name not in metrics_summary:
        metrics_summary[model_name] = {}
        
    metrics_summary[model_name][dataset_type] = {
        "accuracy": record["accuracy"],
        "precision": record["precision"],
        "recall": record["recall"],
        "f1_score": record["f1_score"]
    }

app = FastAPI()

template = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

async def tmp_save_file(upload_file: UploadFile) -> Path:
    try:
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Generate Temporary file path
        file_path = temp_dir / f"{upload_file.filename}"
        
        # Read the contents
        contents = await upload_file.read()
        
        # Write the temporary file
        with open(file_path, "wb") as f:
            f.write(contents)
            
        return file_path

    except Exception as e:
        raise ValueError(f"Error when saving the file: {str(e)}")


@app.get("/")
async def read_root(request: Request):
    return template.TemplateResponse("dashboard.html", {
        "request": request,
        "data_info": data_info,
        "columns_info": columns_info,
        "model_info": model_info,
        "metrics_summary": metrics_summary,
        "plot_image_path": "static/model_comparison.png"

    })

@app.post("/upload-and-evaluate/")
async def upload_and_evaluate(request: Request, file: Optional[UploadFile] = None):
    try:

        if file is None:
            return template.TemplateResponse("dashboard.html", {
                "request": request,
                "data_info": data_info,
                "columns_info": columns_info,
                "model_info": model_info,
                "metrics_summary": metrics_summary,
                 "plot_image_path": "static/model_comparison.png"

            })
        

        # Need HTML to design the layout to upload the file
        temp_file_path = await tmp_save_file(file)

        df_upload = pd.read_csv(temp_file_path)
        
        X_train_upload, X_test_upload, y_train_upload, y_test_upload = preprocess_data(temp_file_path)
        
        
        results_upload, model_comparison_upload = evaluate_models(models, X_train_upload, X_test_upload, y_train_upload, y_test_upload )

        png_path_upload = "static/model_comparison_upload.png"
        model_comparison_upload.savefig(png_path_upload)  

        # Need HTML to show the plot for model_comparison_upload

        data_upload = df_upload.iloc[:5].to_html(classes='table table-striped table-bordered', index=False)# Over 5 records, the layout will be messy
        
        
        columns_upload_df = pd.DataFrame({'Column Name': df_upload.columns.tolist()})
        columns_info_upload = columns_upload_df.to_html(classes='table table-striped table-bordered', index=False)
        
        model_info_upload = {"Show the model info": {}}
        for model_name, model in models.items():
            model_info_upload["Show the model info"][model_name] = model.__class__.__name__

        results_dict_upload = results_upload.to_dict(orient="records")
        metrics_summary_upload = {}
        
        
        for record in results_dict_upload:
            model_name = record["Model"]
            dataset_type = record["Dataset"]
            
            if model_name not in metrics_summary_upload:
                metrics_summary_upload[model_name] = {}
                
            metrics_summary_upload[model_name][dataset_type] = {
                "accuracy": record["accuracy"],
                "precision": record["precision"],
                "recall": record["recall"],
                "f1_score": record["f1_score"]
            }
        
        return template.TemplateResponse("dashboard.html", {
            "request": request,
            "data_info": data_upload,
            "columns_info": columns_info_upload,
            "model_info": model_info_upload,
            "metrics_summary": metrics_summary_upload,
            "plot_image_path": "static/model_comparison_upload.png"

    })

    except ValueError as e:
        return {"error": str(e)}
    
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}
    
