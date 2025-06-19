# image_processor.py
import cv2
import numpy as np
import pytesseract
import pandas as pd
import json
import logging
import easyocr
import os

# लॉगिंग सेटअप
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Tesseract पाथ (Render के लिए)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# EasyOCR रीडर (सिंगलटॉन)
reader = None
def get_easyocr_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en', 'hi'])
    return reader

# इमेज प्री-प्रोसेसिंग
def preprocess_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray = cv2.equalizeHist(gray)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = thresh.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        thresh = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        logging.info("Image preprocessed successfully")
        return thresh
    except Exception as e:
        logging.error(f"Preprocessing error: {str(e)}")
        raise e

# टेबल बॉर्डर डिटेक्शन
def detect_table_borders(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        return lines is not None
    except Exception as e:
        logging.error(f"Table border detection error: {str(e)}")
        return False

# Tesseract से टेबल एक्सट्रैक्शन
def extract_table_tesseract(image):
    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang='eng+hin')
        texts = data['text']
        confs = data['conf']
        lefts = data['left']
        tops = data['top']
        
        table_data = []
        current_row = []
        last_top = tops[0] if tops else 0
        
        for i in range(len(texts)):
            if int(confs[i]) > 60 and texts[i].strip():
                if abs(tops[i] - last_top) > 20:
                    if current_row:
                        table_data.append(current_row)
                    current_row = [texts[i]]
                    last_top = tops[i]
                else:
                    current_row.append(texts[i])
        
        if current_row:
            table_data.append(current_row)
        
        if table_data:
            max_cols = max(len(row) for row in table_data)
            table_data = [row + [''] * (max_cols - len(row)) for row in table_data]
            df = pd.DataFrame(table_data)
            logging.info("Table extracted successfully with Tesseract")
            return df
        else:
            logging.warning("No table data found with Tesseract")
            return None
    except Exception as e:
        logging.error(f"Tesseract table extraction error: {str(e)}")
        return None

# EasyOCR से टेक्स्ट एक्सट्रैक्शन
def extract_text_easyocr(image_path):
    try:
        reader = get_easyocr_reader()
        result = reader.readtext(image_path)
        text = "\n".join([res[1] for res in result])
        logging.info("Text extracted successfully with EasyOCR")
        return text
    except Exception as e:
        logging.error(f"EasyOCR error: {str(e)}")
        return None

# मेन प्रोसेसिंग फंक्शन
def process_image(image_path, output_dir='static/uploads'):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        has_borders = detect_table_borders(img.copy())
        processed_img = preprocess_image(img)
        
        tesseract_text = pytesseract.image_to_string(processed_img, lang='eng+hin', config='--psm 6')
        tesseract_table_df = extract_table_tesseract(processed_img)
        easyocr_text = extract_text_easyocr(image_path)
        
        json_output = {
            "tesseract_text": tesseract_text,
            "easyocr_text": easyocr_text,
            "table_data": tesseract_table_df.to_dict() if tesseract_table_df is not None else {},
            "has_table_borders": has_borders
        }
        json_path = os.path.join(output_dir, 'output.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4)
        
        excel_path = None
        if tesseract_table_df is not None:
            excel_path = os.path.join(output_dir, 'output.xlsx')
            tesseract_table_df.to_excel(excel_path, index=False)
            logging.info(f"Excel file saved: {excel_path}")
        
        return tesseract_text, easyocr_text, tesseract_table_df, json_path, excel_path
    except Exception as e:
        logging.error(f"Processing error: {str(e)}")
        return None, None, None, None, None
