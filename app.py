# app.py
from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
import os
import pandas as pd
from image_processor import process_image

app = Flask(__name__)

# कॉन्फिगरेशन
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# फोल्डर बनाएं
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            tesseract_text, easyocr_text, table_df, json_path, excel_path = process_image(file_path)
            
            if tesseract_text is None:
                return render_template('index.html', error='Error processing image')
            
            table_html = table_df.to_html(index=False) if table_df is not None else None
            
            return render_template('index.html',
                                 tesseract_text=tesseract_text,
                                 easyocr_text=easyocr_text,
                                 table_html=table_html,
                                 image_url=url_for('static', filename=f'uploads/{filename}'),
                                 json_url=url_for('static', filename='uploads/output.json'),
                                 excel_url=url_for('static', filename='uploads/output.xlsx') if excel_path else None)
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
