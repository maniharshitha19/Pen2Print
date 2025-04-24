import os
import re
import sqlite3
import cv2
import numpy as np
import pytesseract
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from textblob import TextBlob
from gtts import gTTS
from fpdf import FPDF
import pdfplumber
import pdf2image
import base64
from PIL import Image
import logging
from datetime import datetime
import tempfile

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Pen2Print')

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata/'

# Database setup
def init_db():
    try:
        conn = sqlite3.connect('pen2print.db')
        c = conn.cursor()
       
        c.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
       
        c.execute('''CREATE TABLE IF NOT EXISTS documents
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    original_text TEXT,
                    corrected_text TEXT,
                    audio_path TEXT,
                    pdf_path TEXT,
                    processing_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id))''')
       
        conn.commit()
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise
    finally:
        conn.close()

init_db()

# Validation functions
def validate_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email)

def validate_password(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    return True, ""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Image processing functions
def preprocess_image(image):
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(opening, -1, kernel)
        return sharpened
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        return image

def correct_text(text):
    try:
        if not text:
            return ""
            
        text = re.sub(r'[^\w\s\'",.?!\-:;()\n]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        replacements = {
            r'\b([|\\/])\b': 'I',
            r'\b¢\b': 'c',
            r'\b©\b': 'c',
            r'\b`\b': "'",
            r'\b“\b': '"',
            r'\b”\b': '"',
            r'\b‘\b': "'",
            r'\b’\b': "'",
            r'\b—\b': '-',
            r'\b–\b': '-',
            r'\b…\b': '...'
        }
       
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
       
        blob = TextBlob(text)
        corrected = str(blob.correct())
        corrected = re.sub(r'([.,!?])([^\s])', r'\1 \2', corrected)
        corrected = re.sub(r'(\s)([.,!?])', r'\2', corrected)
       
        sentences = re.split(r'([.!?] )', corrected)
        corrected = ''.join([s.capitalize() if i % 2 == 0 else s for i, s in enumerate(sentences)])
       
        return corrected
    except Exception as e:
        logger.error(f"Text correction failed: {str(e)}")
        return text

# Enhanced PDF processing with multiple fallbacks
def process_pdf_file(pdf_path):
    start_time = datetime.now()
    full_text = ""
    
    try:
        # Method 1: Try pdfplumber for text-based PDFs
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=1)
                    if page_text:
                        full_text += page_text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed: {str(e)}")
        
        # Method 2: If no text, try OCR on rendered pages
        if not full_text.strip():
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    images = pdf2image.convert_from_path(
                        pdf_path,
                        output_folder=temp_dir,
                        fmt='png',
                        thread_count=4
                    )
                    for i, image in enumerate(images):
                        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                        processed_img = preprocess_image(img)
                        text = pytesseract.image_to_string(
                            processed_img,
                            config='--psm 6 --oem 3 -c preserve_interword_spaces=1'
                        )
                        full_text += text + "\n"
            except Exception as e:
                logger.error(f"PDF OCR failed: {str(e)}")
                raise ValueError("Could not extract text from PDF")
        
        if not full_text.strip():
            raise ValueError("No text could be extracted from PDF")
            
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        full_text = re.sub(r'([.,!?])([^\s])', r'\1 \2', full_text)
        corrected_text = correct_text(full_text)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return full_text, corrected_text, processing_time
    except Exception as e:
        logger.error(f"PDF processing failed: {str(e)}")
        raise

def process_image_file(image_path):
    start_time = datetime.now()
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image file")
       
        height, width = img.shape[:2]
        max_dim = 2500
        if height > max_dim or width > max_dim:
            scale = max_dim / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
       
        processed_img = preprocess_image(img)
        custom_config = r'--oem 3 --psm 6 -l eng'
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        text = re.sub(r'[^\w\s\'",.?!\-:;()\n]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text.strip():
            raise ValueError("No text could be extracted from image")
            
        corrected_text = correct_text(text)
        processing_time = (datetime.now() - start_time).total_seconds()
        return text, corrected_text, processing_time
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise

def text_to_speech(text, output_path):
    try:
        if not text.strip():
            raise ValueError("Empty text for TTS")
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_path)
    except Exception as e:
        logger.error(f"Text-to-speech failed: {str(e)}")
        raise

def text_to_pdf(text, output_path):
    try:
        if not text.strip():
            raise ValueError("Empty text for PDF")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        pdf.output(output_path)
    except Exception as e:
        logger.error(f"PDF generation failed: {str(e)}")
        raise

def get_db_connection():
    conn = sqlite3.connect('pen2print.db')
    conn.row_factory = sqlite3.Row
    return conn

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if len(username) < 4:
            flash('Username must be at least 4 characters', 'danger')
            return redirect(url_for('register'))

        if not validate_email(email):
            flash('Invalid email format', 'danger')
            return redirect(url_for('register'))

        valid_pass, msg = validate_password(password)
        if not valid_pass:
            flash(msg, 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))

        conn = None
        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                       (username, email, password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists', 'danger')
        except Exception as e:
            logger.error(f"Registration failed: {str(e)}")
            flash('Registration failed', 'danger')
        finally:
            if conn:
                conn.close()

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
       
        conn = None
        try:
            conn = get_db_connection()
            user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?",
                              (username, password)).fetchone()
           
            if user:
                session['user_id'] = user['id']
                session['username'] = user['username']
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials', 'danger')
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            flash('Login failed', 'danger')
        finally:
            if conn:
                conn.close()
   
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login first', 'danger')
        return redirect(url_for('login'))
   
    conn = None
    try:
        conn = get_db_connection()
        documents = conn.execute("SELECT id, filename, timestamp FROM documents WHERE user_id = ? ORDER BY timestamp DESC",
                               (session['user_id'],)).fetchall()
        return render_template('dashboard.html', documents=documents)
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        flash('Error loading documents', 'danger')
        return redirect(url_for('home'))
    finally:
        if conn:
            conn.close()

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user_id' not in session:
        flash('Please login first', 'danger')
        return redirect(url_for('login'))
   
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)
       
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)
       
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
           
            conn = None
            try:
                if filename.lower().endswith('.pdf'):
                    text, corrected_text, proc_time = process_pdf_file(filepath)
                else:
                    text, corrected_text, proc_time = process_image_file(filepath)
               
                base_name = os.path.splitext(filename)[0]
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                audio_filename = f"audio_{session['user_id']}_{base_name}_{timestamp}.mp3"
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
                text_to_speech(corrected_text, audio_path)
               
                pdf_filename = f"output_{session['user_id']}_{base_name}_{timestamp}.pdf"
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
                text_to_pdf(corrected_text, pdf_path)
               
                conn = get_db_connection()
                conn.execute('''INSERT INTO documents
                              (user_id, filename, original_text, corrected_text, audio_path, pdf_path, processing_time)
                              VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           (session['user_id'], filename, text, corrected_text, audio_path, pdf_path, proc_time))
                conn.commit()
                doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
               
                return redirect(url_for('result', doc_id=doc_id))
            except Exception as e:
                logger.error(f"Upload processing failed: {str(e)}")
                flash(f'Error processing file: {str(e)}', 'danger')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(url_for('upload'))
            finally:
                if conn:
                    conn.close()
        else:
            flash('Allowed file types: PNG, JPG, JPEG, PDF', 'danger')
   
    return render_template('upload.html')

@app.route('/result/<int:doc_id>', methods=['GET', 'POST'])
def result(doc_id):
    if 'user_id' not in session:
        flash('Please login first', 'danger')
        return redirect(url_for('login'))
   
    conn = None
    try:
        conn = get_db_connection()
        document = conn.execute("SELECT * FROM documents WHERE id = ? AND user_id = ?",
                              (doc_id, session['user_id'])).fetchone()
       
        if not document:
            flash('Document not found', 'danger')
            return redirect(url_for('dashboard'))
       
        if request.method == 'POST':
            corrected_text = request.form['corrected_text']
            base_name = os.path.splitext(document['filename'])[0]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
           
            try:
                audio_filename = f"audio_{session['user_id']}_{base_name}_{timestamp}.mp3"
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
                text_to_speech(corrected_text, audio_path)
               
                pdf_filename = f"output_{session['user_id']}_{base_name}_{timestamp}.pdf"
                pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
                text_to_pdf(corrected_text, pdf_path)
               
                conn.execute('''UPDATE documents
                              SET corrected_text = ?, audio_path = ?, pdf_path = ?
                              WHERE id = ?''',
                           (corrected_text, audio_path, pdf_path, doc_id))
                conn.commit()
               
                try:
                    if document['audio_path'] and os.path.exists(document['audio_path']):
                        os.remove(document['audio_path'])
                    if document['pdf_path'] and os.path.exists(document['pdf_path']):
                        os.remove(document['pdf_path'])
                except Exception as e:
                    logger.error(f"Error deleting old files: {str(e)}")
               
                document = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
               
                flash('Document updated successfully!', 'success')
            except Exception as e:
                logger.error(f"Document update failed: {str(e)}")
                flash('Error updating document', 'danger')
       
        preview_data = None
        if document['filename'].lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                with open(os.path.join(app.config['UPLOAD_FOLDER'], document['filename']), 'rb') as img_file:
                    preview_data = base64.b64encode(img_file.read()).decode('utf-8')
            except Exception as e:
                logger.error(f"Preview image error: {str(e)}")
       
        return render_template('result.html', document=document, preview_data=preview_data)
    except Exception as e:
        logger.error(f"Result page error: {str(e)}")
        flash('Error loading document', 'danger')
        return redirect(url_for('dashboard'))
    finally:
        if conn:
            conn.close()

@app.route('/regenerate_audio/<int:doc_id>', methods=['POST'])
def regenerate_audio(doc_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Please login first'}), 401
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'success': False, 'error': 'No text provided'}), 400
    
    conn = None
    try:
        conn = get_db_connection()
        document = conn.execute("SELECT * FROM documents WHERE id = ? AND user_id = ?",
                              (doc_id, session['user_id'])).fetchone()
        
        if not document:
            return jsonify({'success': False, 'error': 'Document not found'}), 404
        
        base_name = os.path.splitext(document['filename'])[0]
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        audio_filename = f"audio_{session['user_id']}_{base_name}_{timestamp}.mp3"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        
        text_to_speech(data['text'], audio_path)
        
        conn.execute("UPDATE documents SET audio_path = ? WHERE id = ?",
                   (audio_path, doc_id))
        conn.commit()
        
        try:
            if document['audio_path'] and os.path.exists(document['audio_path']):
                os.remove(document['audio_path'])
        except Exception as e:
            logger.error(f"Error deleting old audio file: {str(e)}")
        
        return jsonify({
            'success': True,
            'audio_url': url_for('download_audio', doc_id=doc_id)
        })
    except Exception as e:
        logger.error(f"Audio regeneration failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if conn:
            conn.close()

@app.route('/download_pdf/<int:doc_id>')
def download_pdf(doc_id):
    if 'user_id' not in session:
        flash('Please login first', 'danger')
        return redirect(url_for('login'))
   
    conn = None
    try:
        conn = get_db_connection()
        pdf_path = conn.execute("SELECT pdf_path FROM documents WHERE id = ? AND user_id = ?",
                              (doc_id, session['user_id'])).fetchone()
       
        if not pdf_path:
            flash('PDF not found', 'danger')
            return redirect(url_for('dashboard'))
       
        return send_from_directory(app.config['UPLOAD_FOLDER'],
                                os.path.basename(pdf_path['pdf_path']),
                                as_attachment=True)
    except Exception as e:
        logger.error(f"PDF download failed: {str(e)}")
        flash('Error downloading PDF', 'danger')
        return redirect(url_for('dashboard'))
    finally:
        if conn:
            conn.close()

@app.route('/download_audio/<int:doc_id>')
def download_audio(doc_id):
    if 'user_id' not in session:
        flash('Please login first', 'danger')
        return redirect(url_for('login'))
   
    conn = None
    try:
        conn = get_db_connection()
        audio_path = conn.execute("SELECT audio_path FROM documents WHERE id = ? AND user_id = ?",
                                (doc_id, session['user_id'])).fetchone()
       
        if not audio_path:
            flash('Audio file not found', 'danger')
            return redirect(url_for('dashboard'))
       
        return send_from_directory(app.config['UPLOAD_FOLDER'],
                                os.path.basename(audio_path['audio_path']),
                                as_attachment=True)
    except Exception as e:
        logger.error(f"Audio download failed: {str(e)}")
        flash('Error downloading audio', 'danger')
        return redirect(url_for('dashboard'))
    finally:
        if conn:
            conn.close()

@app.route('/delete/<int:doc_id>')
def delete(doc_id):
    if 'user_id' not in session:
        flash('Please login first', 'danger')
        return redirect(url_for('login'))
   
    conn = None
    try:
        conn = get_db_connection()
        doc_info = conn.execute("SELECT filename, audio_path, pdf_path FROM documents WHERE id = ? AND user_id = ?",
                              (doc_id, session['user_id'])).fetchone()
       
        if doc_info:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
           
            try:
                if doc_info['audio_path'] and os.path.exists(doc_info['audio_path']):
                    os.remove(doc_info['audio_path'])
                if doc_info['pdf_path'] and os.path.exists(doc_info['pdf_path']):
                    os.remove(doc_info['pdf_path'])
                original_file = os.path.join(app.config['UPLOAD_FOLDER'], doc_info['filename'])
                if os.path.exists(original_file):
                    os.remove(original_file)
            except Exception as e:
                logger.error(f"Error deleting old files: {str(e)}")
       
        flash('Document deleted successfully', 'success')
        return redirect(url_for('dashboard'))
    except Exception as e:
        logger.error(f"Document deletion failed: {str(e)}")
        flash('Error deleting document', 'danger')
        return redirect(url_for('dashboard'))
    finally:
        if conn:
            conn.close()

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
