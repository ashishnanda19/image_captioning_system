# Quick Start Guide

Get up and running with the Image Captioning project in 3 simple steps!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First installation may take 5-10 minutes as it downloads PyTorch and the BLIP model.

## Step 2: Choose Your Interface

### Option A: Web Interface (Easiest)

```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

### Option B: Command Line

```bash
python image_captioner.py your_image.jpg
```

### Option C: Python Script

```python
from image_captioner import ImageCaptioner

captioner = ImageCaptioner()
caption = captioner.generate_caption("your_image.jpg")
print(caption)
```

## Step 3: Enjoy!

Upload an image and get AI-generated captions instantly!

## Troubleshooting

**Model download slow?** 
- The model (~990MB) downloads on first run and is cached locally
- Subsequent runs are much faster

**Out of memory?**
- Close other applications
- Process images one at a time

**Need help?**
- Check the full README.md for detailed documentation

---

**That's it! You're ready to caption images! 🎉**

