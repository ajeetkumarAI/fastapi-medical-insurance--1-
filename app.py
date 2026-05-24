import json
import pickle

import numpy as np
import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Load model and metadata once at startup
with open("Linear_Model.pkl", "rb") as f:
    model = pickle.load(f)

with open("project_data.json", "r") as f:
    project_data = json.load(f)

app = FastAPI(title="Medical Insurance Charges Prediction")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

VALID_SEX = {"male", "female"}
VALID_SMOKER = {"yes", "no"}
VALID_REGION = {"southwest", "southeast", "northwest", "northeast"}


def predict_insurance(age, sex, bmi, children, smoker, region):
    test_array = np.zeros(model.n_features_in_)
    test_array[0] = age
    test_array[1] = project_data["sex"][sex]
    test_array[2] = bmi
    test_array[3] = children
    test_array[4] = project_data["smoker"][smoker]
    region_col = f"region_{region}"
    region_index = project_data["columns"].index(region_col)
    test_array[region_index] = 1
    return round(model.predict([test_array])[0], 2)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(name="index.html", request=request)


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    children: float = Form(...),
    smoker: str = Form(...),
    region: str = Form(...),
):
    try:
        sex = sex.strip().lower()
        smoker = smoker.strip().lower()
        region = region.strip().lower()

        if sex not in VALID_SEX:
            return templates.TemplateResponse(name="index.html", request=request, context={"error": "Invalid sex value."})
        if smoker not in VALID_SMOKER:
            return templates.TemplateResponse(name="index.html", request=request, context={"error": "Invalid smoker value."})
        if region not in VALID_REGION:
            return templates.TemplateResponse(name="index.html", request=request, context={"error": "Invalid region value."})
        if age < 0 or age > 150:
            return templates.TemplateResponse(name="index.html", request=request, context={"error": "Age must be between 0 and 150."})
        if bmi < 0:
            return templates.TemplateResponse(name="index.html", request=request, context={"error": "BMI must be non-negative."})
        if children < 0:
            return templates.TemplateResponse(name="index.html", request=request, context={"error": "Children must be non-negative."})

        charges = predict_insurance(age, sex, bmi, children, smoker, region)
        return templates.TemplateResponse(name="index.html", request=request, context={"charges": charges})

    except Exception as e:
        return templates.TemplateResponse(name="index.html", request=request, context={"error": f"Invalid input: {e}"})


@app.get("/health")
async def health():
    return JSONResponse({"status": "healthy"})


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
