import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import joblib
import uvicorn

# Initialize FastAPI app
app = FastAPI()

    
# Load pre-trained pipeline and model with error handling
try:
    with open("pipeline.pkl", 'rb') as f:
        loaded_pipeline = pickle.load(f)
    with open("gradient_boosting_model.joblib.dat", 'rb') as model:
        loaded_model = joblib.load(model)
except (FileNotFoundError, IOError) as e:
    raise RuntimeError("Model or pipeline file is missing or corrupt") from e

# Define the data model for incoming requests
class RequestData(BaseModel):
    hour: int
    traffic_partner: str
    position: int
    os_name: str
    gender_impute: str
    age_impute: int
    householdincome_impute: int
    campaign_name: str
    campagin_id: str

@app.post("/test")
async def predict_ctr(request_data: RequestData):
    """
    Receives campaign data and returns the predicted Click-Through Rate (CTR).
    """
    try:
        # Convert request data to DataFrame
        df_test = pd.DataFrame([request_data.dict()])
        model_features = ['hour', 'position', 'age_impute', 'householdincome_impute', 
                          'traffic_partner', 'os_name', 'gender_impute', 'campaign_name']

        # Data transformation and prediction
        data_processed = loaded_pipeline.transform(df_test[model_features])
        pred_prob = loaded_model.predict_proba(data_processed)

        # Prepare response DataFrame
        df_test['model_ctr'] = pred_prob[:, 1]
        result_df = df_test[['campaign_name', 'position', 'model_ctr']]

        # Return JSON response
        return result_df.to_dict(orient="records")

    except Exception as e:
        error_message = f"An error occurred during prediction: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
