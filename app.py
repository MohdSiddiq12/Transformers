from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import functools

app = Flask(__name__)

# Load pre-trained model and tokenizer once when the application starts
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

@functools.lru_cache(maxsize=128)
def predict_next_word(input_sentence):
    input_ids = tokenizer.encode(input_sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(input_ids)
    next_token_logits = outputs.logits[:, -1, :]
    probabilities = torch.softmax(next_token_logits, dim=-1)
    predicted_token_id = torch.argmax(probabilities, dim=-1)
    predicted_word = tokenizer.decode(predicted_token_id)
    return predicted_word

@app.route('/predict', methods=['POST'])
def predict():
    input_sentence = request.form['sentence']
    predicted_word = predict_next_word(input_sentence)
    return jsonify({'next_word': predicted_word})

if __name__ == '__main__':
    app.run(debug=True)
