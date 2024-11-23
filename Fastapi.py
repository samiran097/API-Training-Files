
import pickle
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse 
import io
import uvicorn

pick_file = "pipeline.pkl"
with open(pick_file, 'rb') as f:
    loaded_pipeline = pickle.load(f)

with open('gradient_boosting_model.joblib.dat', 'rb') as model:
    loaded_model = joblib.load(model)

app = FastAPI()

@app.post("/test")
async def root(file: UploadFile = File(...)):
    """
    Endpoint to make predictions based on uploaded CSV file.
    
    - **test**: CSV file with required model features.
    - Returns: JSON response with predictions.
    """
    print("File is received")
    try:
        contents = await file.read()  # Read the contents of the uploaded file
        file_like_object = io.BytesIO(contents)  # Convert bytes to file-like object
        df_test = pd.read_csv(file_like_object) # Read CSV contents into DataFrame
        print("Data read successfully.")
        data = df_test.copy()

        model_features = ['hour', 'position', 'age_impute', 'householdincome_impute',
                          'traffic_partner', 'os_name', 'gender_impute', 'campaign_name']

        data.reset_index(drop=True, inplace=True)
        data_processed = loaded_pipeline.transform(data[model_features])
        pred_prob = loaded_model.predict_proba(data_processed)
        print("Prediction probabilities:", pred_prob)

        data['model_ctr'] = pred_prob[:, 1]
        opfeature = ['campagin_id', 'position', 'model_ctr']
        final = data[opfeature]


        return final.to_json(orient="records")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

###########



# import pickle
# import pandas as pd
# import joblib
# from fastapi import FastAPI, UploadFile, File, HTTPException
# import io
# import uvicorn

# # Constants for file paths
# PIPELINE_FILE = "pipeline.pkl"
# MODEL_FILE = "gradient_boosting_model.joblib.dat"

# def load_pipeline(file_path):
#     """Load and return the preprocessing pipeline from a pickle file."""
#     try:
#         with open(file_path, 'rb') as f:
#             return pickle.load(f)
#     except Exception as e:
#         raise RuntimeError(f"Error loading pipeline: {e}")

# def load_model(file_path):
#     """Load and return the model from a joblib file."""
#     try:
#         with open(file_path, 'rb') as model_file:
#             return joblib.load(model_file)
#     except Exception as e:
#         raise RuntimeError(f"Error loading model: {e}")

# # Load the pipeline and model
# loaded_pipeline = load_pipeline(PIPELINE_FILE)
# loaded_model = load_model(MODEL_FILE)

# # Initialize FastAPI app
# app = FastAPI()

# @app.post("/predict")
# async def predict(test: UploadFile = File(...)):
#     """
#     Endpoint to make predictions based on uploaded CSV file.
    
#     - **test**: CSV file with required model features.
#     - Returns: JSON response with predictions.
#     """
#     print("File received for prediction.")
#     try:
#         # Read the contents of the uploaded file
#         contents = await test.read()
#         file_like_object = io.BytesIO(contents)  # Convert bytes to file-like object
#         df_test = pd.read_csv(file_like_object)
#         print("Data successfully read into DataFrame.")

#         # Define required model features
#         model_features = [
#             'hour', 'position', 'age_impute', 'householdincome_impute', 
#             'traffic_partner', 'os_name', 'gender_impute', 'campaign_name'
#         ]
        
#         # Verify that all required features are present
#         if not set(model_features).issubset(df_test.columns):
#             missing_features = list(set(model_features) - set(df_test.columns))
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Missing required features: {missing_features}"
#             )
        
#         # Process data and make predictions
#         data_processed = loaded_pipeline.transform(df_test[model_features])
#         pred_prob = loaded_model.predict_proba(data_processed)
#         print("Prediction probabilities calculated.")

#         # Append predictions to DataFrame
#         df_test['model_ctr'] = pred_prob[:, 1]
#         output_features = ['campaign_id', 'position', 'model_ctr']

#         # Check if output features are in the DataFrame
#         if not set(output_features).issubset(df_test.columns):
#             raise HTTPException(
#                 status_code=400,
#                 detail="Output features missing in the input data."
#             )

#         result = df_test[output_features]
#         print("Returning prediction results.")
#         return result.to_json(orient="records")

#     except pd.errors.EmptyDataError:
#         raise HTTPException(status_code=400, detail="Uploaded file is empty.")
#     except Exception as e:
#         print(f"An error occurred during prediction: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)
