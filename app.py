from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

app = Flask(__name__)

# Load pre-trained model and tokenizer once when the application starts
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def predict_next_word(input_sentence):
    try:
        input_ids = tokenizer.encode(input_sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1)
        predicted_token_id = torch.argmax(probabilities, dim=-1)
        predicted_word = tokenizer.decode(predicted_token_id)
        return predicted_word
    except Exception as e:
        return str(e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_sentence = request.json.get('sentence')
        if not input_sentence:
            return jsonify({'error': 'No input sentence provided'}), 400
        predicted_word = predict_next_word(input_sentence)
        return jsonify({'next_word': predicted_word})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
