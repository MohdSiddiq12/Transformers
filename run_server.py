from waitress import serve
import app  # Assuming your main app script is named `app.py`

if __name__ == "__main__":
    serve(app.app, host="0.0.0.0", port=8000)
