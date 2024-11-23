import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import json
import pickle
import joblib

pick_file ="pipeline.pkl"
with open(pick_file, 'rb') as f:
    loaded_pipeline = pickle.load(f)

with open('gradient_boosting_model.joblib.dat', 'rb') as model:
    loaded_model = joblib.load(model)


singleapp = Flask(__name__)
@singleapp.route('/test', methods = ['POST'])
def handle():
    """
    Endpoint to make predictions based on input data.
    
    Expects a CSV file with the necessary features.
    Returns a JSON response containing predictions.
    """

    if not request.json:
        return jsonify({"error": "No values part in the request"}), 400
    
    if request.json:
        data = request.json
        print(data)
        df_test = pd.DataFrame([data])
        print(df_test)
        df=df_test.copy()

    model_features = ['hour', 'position', 'age_impute', 'householdincome_impute', 
                  'traffic_partner', 'os_name', 'gender_impute', 'campaign_name']
    
    try:
        data_processed = loaded_pipeline.transform(df[model_features])
        pred_prob = loaded_model.predict_proba(data_processed)
        print(pred_prob)

        
        df['model_ctr'] = pred_prob[:, 1]
        opfeature = ['campaign_name', 'position', 'model_ctr']
        final = df[opfeature]
        print(final)
        print(type(final))

        return final.to_json(orient="records")
        # return data, df.to_json(orient='records')
    except Exception as e:
        print("An error occurred:", str(e))
        return {"error": str(e)}
    
if __name__ == '__main__':
    singleapp.run(debug=True,threaded=True)


# [0.95885, 0.05225]

# pred = 0, 1