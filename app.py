"""
Flask Web Application for Image Captioning
Provides a simple web interface to upload images and get captions
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from image_captioner import ImageCaptioner
import os
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the captioner (loads model on startup)
print("Initializing image captioner...")
captioner = ImageCaptioner()
print("Ready to caption images!")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/caption', methods=['POST'])
def caption_image():
    """Handle image upload and generate caption"""
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP, WEBP'}), 400
        
        # Read image from request
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Generate caption
        caption = captioner.generate_caption_from_pil(image)
        
        return jsonify({
            'success': True,
            'caption': caption,
            'filename': secure_filename(file.filename)
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'loaded'})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Textify - An Image to Text Converter")
    print("="*50)
    print("Access the app at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

