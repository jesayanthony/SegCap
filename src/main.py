from segmentation import SegmentAndCaption
import os

def main():
    try:
        # Print current directory
        print("Current directory:", os.getcwd())
        
        # Configure paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "sam_vit_h_4b8939.pth")
        input_dir = os.path.join(current_dir, "input_images")
        output_dir = os.path.join(current_dir, "output_images")
        
        # Print paths to verify
        print(f"Model path: {model_path}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        # Verify input directory exists and has images
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found at {input_dir}")
            
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            raise FileNotFoundError(f"No images found in {input_dir}")
        
        print(f"Found images: {image_files}")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the system
        print("Initializing SegmentAndCaption system...")
        segcap = SegmentAndCaption(sam_checkpoint=model_path)
        
        # Process all images
        for image_file in image_files:
            print(f"\nProcessing image: {image_file}")
            image_path = os.path.join(input_dir, image_file)
            try:
                results, _ = segcap.process_image(
                    image_path=image_path,
                    output_dir=output_dir
                )
                print(f"Successfully processed {image_file}")
            except Exception as e:
                print(f"Error processing {image_file}: {str(e)}")
                import traceback
                print(traceback.format_exc())

    except Exception as e:
        print(f"Main error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()