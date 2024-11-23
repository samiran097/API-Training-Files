import pickle
import pandas as pd
import numpy as  np
import json
import joblib
from flask import Flask, request, jsonify

pick_file ="pipeline.pkl"
with open(pick_file, 'rb') as f:
    loaded_pipeline = pickle.load(f)

with open('gradient_boosting_model.joblib.dat', 'rb') as model:
    loaded_model = joblib.load(model)
    
app= Flask(__name__)

@app.route('/test', methods =['post'])
def predict():
    """
    Endpoint to make predictions based on input data.
    
    Expects a CSV file with the necessary features.
    Returns a JSON response containing predictions.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    try: 
    # may be change according to the inputs
        file = request.files['file']
        df_test=pd.read_csv(file)
        data=df_test.copy()
        print(data.head())
    
        
        model_features = ['hour', 'position', 'age_impute', 'householdincome_impute', 
                    'traffic_partner', 'os_name', 'gender_impute', 'campaign_name']
        

        data.reset_index(drop=False, inplace=True)
        data_processed = loaded_pipeline.transform(data[model_features])
        pred_prob = loaded_model.predict_proba(data_processed)
        print(pred_prob)
        data['model_ctr'] = pred_prob[:, 1]
        opfeature = ['campagin_id', 'position', 'model_ctr']
        final = data[opfeature]
        return final.to_json(orient="records")
    
    except Exception as e:
        print("An error occurred:", str(e))
        return {"error": str(e)}



if __name__ == '__main__':
    app.run(debug=True,threaded=True)



