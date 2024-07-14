async function predictNextWord() {
    const inputSentence = document.getElementById('inputSentence').value;
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `sentence=${encodeURIComponent(inputSentence)}`,
    });
    const data = await response.json();
    document.getElementById('output').innerText = `Next word: ${data.next_word}`;
}
