function predictNextWord() {
    const inputSentence = document.getElementById('inputSentence').value;
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `sentence=${encodeURIComponent(inputSentence)}`,
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('output').innerText = `Next word: ${data.next_word}`;
    })
    .catch(error => console.error('Error:', error));
}
