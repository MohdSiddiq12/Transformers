from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
API_KEY = "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"  # Replace with your actual API key
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

def query(payload):
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    return response.json()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_sentence = request.json.get('sentence')
    if not input_sentence:
        return jsonify({'error': 'No input sentence provided'}), 400

    payload = {
        "inputs": input_sentence,
        "parameters": {
            "max_new_tokens": 1,
            "return_full_text": False
        }
    }

    try:
        result = query(payload)
        if isinstance(result, list) and len(result) > 0:
            predicted_text = result[0]['generated_text']
            words = predicted_text.split()
            predicted_word = words[-1] if words else ""
            return jsonify({'next_word': predicted_word})
        else:
            return jsonify({'error': 'Unexpected response format'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
