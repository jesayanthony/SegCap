import torch
from segment_anything import sam_model_registry, SamPredictor
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

class SegmentAndCaption:
    def __init__(self, sam_checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the segmentation and captioning system.
        """
        print(f"Using device: {device}")
        self.device = device
        
        # Initialize SAM
        print("Loading SAM model...")
        self.sam = sam_model_registry["default"](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        
        # Initialize BLIP
        print("Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.captioner = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        print("Models loaded successfully!")

    def segment_image(self, image_path):
        """
        Segment image with improved parameters.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        
        # Calculate image dimensions
        h, w = image.shape[:2]
        
        # Generate more focused points
        points = []
        
        # Create a denser grid of points
        rows, cols = 10, 10
        for i in range(rows):
            for j in range(cols):
                x = int(w * (j + 0.5) / cols)
                y = int(h * (i + 0.5) / rows)
                points.append([x, y])
        
        # Add center region points
        center_x, center_y = w // 2, h // 2
        for dx in [-w//4, 0, w//4]:
            for dy in [-h//4, 0, h//4]:
                points.append([center_x + dx, center_y + dy])
        
        # Get predictions
        all_masks = []
        input_points = np.array(points)
        
        # Process smaller batches of points
        batch_size = 5
        for i in range(0, len(points), batch_size):
            batch_points = input_points[i:i+batch_size]
            batch_labels = np.ones(len(batch_points))
            
            masks, _, _ = self.predictor.predict(
                point_coords=batch_points,
                point_labels=batch_labels,
                multimask_output=True
            )
            all_masks.extend(masks)
        
        # Process segments with better filtering
        segments = []
        min_area = (h * w) * 0.02  # 2% of image
        max_area = (h * w) * 0.7   # 70% of image
        
        for mask in all_masks:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                area = cv2.contourArea(contours[0])
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    if w > 20 and h > 20:  # Minimum size threshold
                        segments.append({
                            'mask': mask,
                            'bbox': (x, y, w, h)
                        })
        
        # Remove overlapping segments
        final_segments = []
        for i, seg1 in enumerate(segments):
            overlap = False
            for seg2 in final_segments:
                intersection = np.logical_and(seg1['mask'], seg2['mask']).sum()
                union = np.logical_or(seg1['mask'], seg2['mask']).sum()
                iou = intersection / union if union > 0 else 0
                
                if iou > 0.3:
                    overlap = True
                    break
            
            if not overlap:
                final_segments.append(seg1)
        
        return final_segments, image

    def caption_segment(self, image, segment):
        """
        Generate accurate captions for image segments.
        """
        try:
            mask = segment['mask']
            masked_image = image.copy()
            masked_image[~mask] = 0  # Use black background for better contrast
            
            x, y, w, h = segment['bbox']
            pad = 10
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_end = min(image.shape[1], x + w + pad)
            y_end = min(image.shape[0], y + h + pad)
            cropped = masked_image[y_start:y_end, x_start:x_end]
            
            pil_image = Image.fromarray(cropped)
            
            # Use a specific prompt for object identification
            inputs = self.processor(
                images=pil_image,
                text="an object: ",  # Direct object prompt
                return_tensors="pt"
            ).to(self.device)
            
            out = self.captioner.generate(
                **inputs,
                max_length=20,
                num_beams=5,
                min_length=5,
                temperature=0.5,  # Lower temperature for more focused description
                do_sample=False,  # Disable sampling for more consistent output
                repetition_penalty=1.5
            )
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            
            # Clean up caption
            unwanted_phrases = [
                "an object: ",
                "a black and white image",
                "a photo of",
                "an image of",
                "a picture of",
                "a close up of",
                "a photograph of",
                "this is",
                "square frame",
                "the background",
                "profile"
            ]
            
            caption = caption.lower()
            for phrase in unwanted_phrases:
                caption = caption.replace(phrase.lower(), "")
            
            caption = " ".join(caption.split())  # Clean up spaces
            caption = caption.strip()
            
            # Capitalize first letter
            if caption:
                caption = caption[0].upper() + caption[1:]
            
            return caption
                
        except Exception as e:
            print(f"Error in caption_segment: {str(e)}")
            return "An object in the image"

    def process_image(self, image_path, output_dir=None):
        """
        Process an image through the full pipeline.
        """
        segments, image = self.segment_image(image_path)
        print(f"\nProcessing image: {os.path.basename(image_path)}")
        print("Found segments:", len(segments))
        
        results = []
        for i, segment in enumerate(segments):
            caption = self.caption_segment(image, segment)
            # Print the caption
            print(f"Caption {i+1}: {caption}")
            
            results.append({
                'segment': segment,
                'caption': caption
            })
            
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, 
                f"result_{os.path.basename(image_path)}"
            )
            self.save_visualization(image, results, output_path)
            print(f"Saved visualization to: {output_path}")
        
        return results, image

    def save_visualization(self, image, results, output_path):
        """
        Save visualization of results.
        """
        plt.figure(figsize=(15, 5 * (len(results) + 1)))
        
        # Show original image
        plt.subplot(len(results) + 1, 1, 1)
        plt.imshow(image)
        plt.title("Original Image", pad=20, fontsize=12)
        plt.axis('off')
        
        # Show each segment and its caption
        for i, result in enumerate(results):
            plt.subplot(len(results) + 1, 1, i + 2)
            
            masked_image = image.copy()
            masked_image[~result['segment']['mask']] = 0  # Black background
            plt.imshow(masked_image)
            plt.title(result['caption'], 
                     pad=20, 
                     fontsize=12,
                     wrap=True)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()