"""
Example usage script for the Image Captioning project
Demonstrates different ways to use the captioner
"""

from image_captioner import ImageCaptioner
from PIL import Image
import os


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize the captioner
    captioner = ImageCaptioner()
    
    # Generate caption for an image
    # Replace with your image path
    image_path = "sample_image.jpg"
    
    if os.path.exists(image_path):
        caption = captioner.generate_caption(image_path)
        print(f"Image: {image_path}")
        print(f"Caption: {caption}\n")
    else:
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path.\n")


def example_custom_parameters():
    """Example with custom parameters"""
    print("=" * 60)
    print("Example 2: Custom Parameters")
    print("=" * 60)
    
    captioner = ImageCaptioner()
    image_path = "sample_image.jpg"
    
    if os.path.exists(image_path):
        # Longer caption with more beams for better quality
        caption = captioner.generate_caption(
            image_path,
            max_length=100,  # Longer captions
            num_beams=5      # More beams = better quality
        )
        print(f"Image: {image_path}")
        print(f"Caption (longer, higher quality): {caption}\n")
    else:
        print(f"Image not found: {image_path}\n")


def example_pil_image():
    """Example using PIL Image object"""
    print("=" * 60)
    print("Example 3: Using PIL Image Object")
    print("=" * 60)
    
    captioner = ImageCaptioner()
    image_path = "sample_image.jpg"
    
    if os.path.exists(image_path):
        # Load image with PIL
        image = Image.open(image_path)
        
        # Generate caption from PIL Image
        caption = captioner.generate_caption_from_pil(image)
        print(f"Image: {image_path}")
        print(f"Caption: {caption}\n")
    else:
        print(f"Image not found: {image_path}\n")


def example_batch_processing():
    """Example for processing multiple images"""
    print("=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)
    
    captioner = ImageCaptioner()
    
    # List of image paths
    image_paths = [
        "image1.jpg",
        "image2.jpg",
        "image3.jpg"
    ]
    
    print("Processing multiple images...\n")
    
    for image_path in image_paths:
        if os.path.exists(image_path):
            caption = captioner.generate_caption(image_path)
            print(f"Image: {image_path}")
            print(f"Caption: {caption}\n")
        else:
            print(f"Skipping {image_path} (not found)\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Image Captioning - Example Usage")
    print("=" * 60 + "\n")
    
    # Run examples
    example_basic_usage()
    example_custom_parameters()
    example_pil_image()
    example_batch_processing()
    
    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nTo use with your own images:")
    print("1. Place your images in the project directory")
    print("2. Update the image_path variable in the examples")
    print("3. Run: python example_usage.py")
    print("\nOr use the web interface:")
    print("Run: python app.py")
    print("Then open: http://localhost:5000")

