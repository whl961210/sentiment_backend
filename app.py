from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your trained model and vectorizer
model = joblib.load('trained_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    print("Received data:", data)
    text = data.get('text')

    if not isinstance(text, str):
        return jsonify({'error': 'Text data is not a string'}), 400

    # Vectorize the text
    processed_text = vectorizer.transform([text])

    # Predict sentiment
    prediction = model.predict(processed_text)[0]
    sentiment = 'Positive' if prediction == "1" else 'Negative'

    return jsonify({'text': text, 'sentiment': sentiment})
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    column_name = request.form.get('column_name', 'text')  # Get column name from the request, default to 'text'

    # Read the file and process
    if file:
        df = pd.read_csv(file)
        if column_name not in df.columns:
            return jsonify({'error': f'Column {column_name} not found in the file'}), 400

        processed_texts = vectorizer.transform(df[column_name])
        predictions = model.predict(processed_texts)
        df['Sentiment'] = ['Positive' if pred == "1" else 'Negative' for pred in predictions]

        # Convert dataframe to a list of dictionaries for JSON response
        results = df.to_dict(orient='records')
        return jsonify(results)

    return jsonify({'error': 'No file selected'}), 400
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
