# Image Captioning System

A complete machine learning project that generates text descriptions for images using a pre-trained BLIP (Bootstrapping Language-Image Pre-training) model. This project runs entirely locally without any API calls to external services.

## Features

- 🖼️ Generate captions for images using state-of-the-art BLIP model
- 🌐 Web interface for easy image upload and caption generation
- 💻 Command-line interface for batch processing
- 🚀 Runs entirely locally - no API calls required
- 🎯 Supports multiple image formats (PNG, JPG, JPEG, GIF, BMP, WEBP)

## Model Information

This project uses the **BLIP (Bootstrapping Language-Image Pre-training)** model from Salesforce, specifically the `blip-image-captioning-base` variant. The model is downloaded automatically from Hugging Face on first run and cached locally.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** The first installation may take a few minutes as it downloads PyTorch and other large dependencies.

## Usage

### Method 1: Web Interface (Recommended)

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

3. **Upload an image** using the web interface and get instant captions!

### Method 2: Command Line Interface

**Single image:**
```bash
python image_captioner.py path/to/your/image.jpg
```

**Example:**
```bash
python image_captioner.py sample_image.jpg
```

### Method 3: Python Script

```python
from image_captioner import ImageCaptioner

# Initialize the captioner
captioner = ImageCaptioner()

# Generate caption for an image
caption = captioner.generate_caption("path/to/image.jpg")
print(f"Caption: {caption}")
```

## Project Structure

```
pbl/
├── image_captioner.py    # Main captioning module
├── app.py                # Flask web application
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/           # HTML templates for web app
│   └── index.html
└── uploads/            # Uploaded images (created automatically)
```

## How It Works

1. **Model Loading**: On first run, the BLIP model is downloaded from Hugging Face and cached locally (~990MB)
2. **Image Processing**: Images are loaded and preprocessed using the BLIP processor
3. **Caption Generation**: The model generates a text description using conditional generation
4. **Output**: The generated caption is returned as a string

## Technical Details

- **Model**: Salesforce/blip-image-captioning-base
- **Framework**: PyTorch + Transformers (Hugging Face)
- **Device**: Automatically uses GPU if available, falls back to CPU
- **Max Caption Length**: 200 tokens (configurable)
- **Beam Search**: Uses 5 beams for better quality (configurable)

## Performance

- **First Run**: ~30-60 seconds (model download and loading)
- **Subsequent Runs**: ~2-5 seconds per image (depending on hardware)
- **GPU Acceleration**: Significantly faster on systems with CUDA-compatible GPUs

## Troubleshooting

### Model Download Issues
If you encounter download issues, the model will be cached in:
- Windows: `C:\Users\<username>\.cache\huggingface\`
- Linux/Mac: `~/.cache/huggingface/`

### Memory Issues
If you run out of memory:
- Close other applications
- Use a smaller model variant (modify `model_name` in `image_captioner.py`)
- Process images one at a time

### CUDA/GPU Issues
The model automatically falls back to CPU if CUDA is not available. To force CPU usage:
```python
captioner = ImageCaptioner()
captioner.device = "cpu"
```

## Customization

### Adjust Caption Length
```python
caption = captioner.generate_caption("image.jpg", max_length=100)
```

### Adjust Quality vs Speed
```python
# Higher beams = better quality, slower (default: 3)
caption = captioner.generate_caption("image.jpg", num_beams=5)
```

## License

This project uses the BLIP model which is licensed under BSD-3-Clause. Please refer to the original model's license for commercial use.

## Credits

- **BLIP Model**: Salesforce Research
- **Transformers Library**: Hugging Face
- **PyTorch**: Facebook AI Research

## Future Enhancements

Potential improvements you could add:
- Batch processing for multiple images
- Caption refinement options
- Different model variants (larger/smaller)
- Image preprocessing options
- Export captions to file
- Multi-language support

## Support

For issues or questions, please check:
1. Ensure all dependencies are installed correctly
2. Verify Python version is 3.8+
3. Check that images are in supported formats
4. Review error messages for specific issues

---

**Enjoy generating image captions! 🎉**
