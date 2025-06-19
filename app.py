# app.py

from flask import Flask, render_template, request
from ocr_engine import process_image
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            path = os.path.join('static', image.filename)
            image.save(path)
            result = process_image(path)
    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
