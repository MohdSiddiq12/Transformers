import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def predict_next_word(input_sentence):
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
            return predicted_word
        else:
            return "Error: Unexpected response format"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_sentence = request.json.get('sentence')
    if not input_sentence:
        return jsonify({'error': 'No input sentence provided'}), 400
    
    predicted_word = predict_next_word(input_sentence)
    return jsonify({'next_word': predicted_word})

if __name__ == '__main__':
    app.run(debug=True)