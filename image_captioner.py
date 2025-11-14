"""
Image Captioning Module
Uses BLIP (Bootstrapping Language-Image Pre-training) model for generating image captions.
This runs entirely locally without any API calls.
"""

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os


class ImageCaptioner:
    """Image Captioning class using BLIP model"""
    
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """
        Initialize the image captioning model
        
        Args:
            model_name: Hugging Face model identifier for BLIP
        """
        print("Loading BLIP model... This may take a moment on first run.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate_caption(self, image_path, max_length=200, num_beams=5, length_penalty=1.2):
        """
        Generate caption for an image
        
        Args:
            image_path: Path to the image file
            max_length: Maximum length of generated caption (increased for more detail)
            num_beams: Number of beams for beam search (higher = better quality, slower)
            length_penalty: Penalty for length (values > 1.0 encourage longer outputs)
        
        Returns:
            str: Generated caption
        """
        try:
            # Load and preprocess image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = Image.open(image_path).convert('RGB')
            
            # Process image
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption with enhanced parameters for detailed output
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_length=max_length, 
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=1.5,  # Prevent repetitive text
                    do_sample=False,  # Use deterministic beam search
                    early_stopping=True
                )
            
            # Decode caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        
        except Exception as e:
            return f"Error generating caption: {str(e)}"
    
    def generate_caption_from_pil(self, image, max_length=200, num_beams=5, length_penalty=1.2):
        """
        Generate caption from PIL Image object
        
        Args:
            image: PIL Image object
            max_length: Maximum length of generated caption (increased for more detail)
            num_beams: Number of beams for beam search (higher = better quality)
            length_penalty: Penalty for length (values > 1.0 encourage longer outputs)
        
        Returns:
            str: Generated caption
        """
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process image with a prompt to encourage detailed descriptions
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption with enhanced parameters for detailed output
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_length=max_length, 
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=1.5,  # Prevent repetitive text
                    do_sample=False,  # Use deterministic beam search
                    early_stopping=True
                )
            
            # Decode caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        
        except Exception as e:
            return f"Error generating caption: {str(e)}"


def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_captioner.py <image_path>")
        print("Example: python image_captioner.py sample_image.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Initialize captioner
    captioner = ImageCaptioner()
    
    # Generate caption
    print(f"\nProcessing image: {image_path}")
    caption = captioner.generate_caption(image_path)
    print(f"\nGenerated Caption: {caption}\n")


if __name__ == "__main__":
    main()

