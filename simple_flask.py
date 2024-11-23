from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the request has a file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    # Check if a file is actually uploaded
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Read the CSV file into a DataFrame
    try:
        df = pd.read_csv(file)
        """Data preperation""" 
        
        df= df[['Date', 'count']]

        """Data end preperation"""
        result = df.to_dict(orient="records")  # Convert to list of dictionaries
        return jsonify(result)  # Convert to JSON and return
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)