from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
STORAGE_DIR = './storage/'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(STORAGE_DIR, filename))
        return jsonify({"message": "file uploaded"}), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/status', methods=['GET'])
def status():
    return "OK!"
