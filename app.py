import requests
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

HUGGING_FACE_API_URL = 'https://api-inference.huggingface.co/models/gpt2'
HUGGING_FACE_API_KEY = ('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')

def predict_next_word(input_sentence):
    headers = {
        'Authorization': f'Bearer {HUGGING_FACE_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'inputs': input_sentence,
        'parameters': {
            'max_new_tokens': 1
        }
    }

    response = requests.post(HUGGING_FACE_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return f"Error: {response.json().get('error', 'Unknown error')}"
    
    result = response.json()
    predicted_word = result[0]['generated_text'].split(input_sentence)[1].strip()
    return predicted_word

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
