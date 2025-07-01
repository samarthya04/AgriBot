from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
import hashlib
import structlog

logger = structlog.get_logger()
upload_bp = Blueprint('upload', __name__)

def validate_image(file):
    if not file or not secure_filename(file.filename):
        return False, "Invalid filename"
    if not file.content_type.startswith('image/'):
        return False, "Invalid file type"
    try:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > app.config['MAX_CONTENT_LENGTH']:
            return False, "File too large"
        Image.open(file).verify()
        file.seek(0)
        return True, "Valid image"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

@upload_bp.route("/api/upload-image", methods=["POST"])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        is_valid, message = validate_image(file)
        if not is_valid:
            return jsonify({"success": False, "error": message}), 400
        
        image_data = file.read()
        image_hash = hashlib.sha256(image_data).hexdigest()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        logger.info("Image uploaded successfully", file_size=len(image_data))
        return jsonify({
            "success": True,
            "image_hash": image_hash,
            "image_data": base64_image,
            "file_size": len(image_data)
        })
    
    except Exception as e:
        logger.error("Image upload error", error=str(e))
        return jsonify({"success": False, "error": "Upload failed"}), 500