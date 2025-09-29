#!/usr/bin/env python3
"""
Visualize YOLOv8 C Model Detections
Reads detection results from C model and overlays them on the original image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# YOLOv8 COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
    'toothbrush'
]

def parse_yolov8_output(output_file="c_detections.txt"):
    """Parse detection results from C model output file"""
    detections = []
    
    try:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if 'Box' in line and ':' in line:
                # Parse detection line format: "Box N: x=X.X, y=Y.Y, w=W.W, h=H.H"
                parts = line.strip().split(':')[1].split(',')
                x = float(parts[0].split('=')[1])
                y = float(parts[1].split('=')[1])
                w = float(parts[2].split('=')[1])
                h = float(parts[3].split('=')[1])
                
                detections.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'confidence': 0.5,  # Default confidence
                    'class_id': 0,      # Default to 'person'
                    'class_name': 'detection'
                })
    except FileNotFoundError:
        print(f"Detection file {output_file} not found. Using sample detections.")
        # Sample detections from the test output
        sample_detections = [
            {'x': 3.516, 'y': 8.871, 'w': 7.141, 'h': 17.904},
            {'x': 22.497, 'y': 5.520, 'w': 45.067, 'h': 11.102},
            {'x': 33.424, 'y': 3.832, 'w': 66.151, 'h': 7.732},
            {'x': 51.997, 'y': 3.398, 'w': 101.728, 'h': 6.844},
            {'x': 56.455, 'y': 3.568, 'w': 108.787, 'h': 7.160}
        ]
        
        for i, det in enumerate(sample_detections):
            detections.append({
                'x': det['x'], 'y': det['y'], 'w': det['w'], 'h': det['h'],
                'confidence': 0.7 - i*0.1,  # Decreasing confidence
                'class_id': 0,
                'class_name': f'detection_{i+1}'
            })
    
    return detections

def load_and_resize_image(image_path="../images/street.ppm", target_size=(640, 640)):
    """Load and resize image to match model input size"""
    try:
        # Try loading as PPM first
        img = cv2.imread(image_path)
        if img is None:
            # Try other formats
            for ext in ['.jpg', '.png']:
                alt_path = image_path.replace('.ppm', ext)
                img = cv2.imread(alt_path)
                if img is not None:
                    image_path = alt_path
                    break
        
        if img is None:
            print(f"Could not load image from {image_path}")
            return None, None, None
            
        original_shape = img.shape[:2]  # (height, width)
        
        # Resize to target size
        resized = cv2.resize(img, target_size)
        
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        return img_rgb, resized_rgb, original_shape
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None

def apply_nms(detections, conf_threshold=0.25, iou_threshold=0.45):
    """Apply Non-Maximum Suppression to filter overlapping boxes"""
    if not detections:
        return []
    
    # Filter by confidence
    filtered = [d for d in detections if d['confidence'] >= conf_threshold]
    
    if not filtered:
        return []
    
    # Convert to format for NMS (x1, y1, x2, y2, confidence)
    boxes = []
    for det in filtered:
        x1 = det['x'] - det['w']/2
        y1 = det['y'] - det['h']/2
        x2 = det['x'] + det['w']/2
        y2 = det['y'] + det['h']/2
        boxes.append([x1, y1, x2, y2, det['confidence']])
    
    boxes = np.array(boxes)
    
    # Simple NMS implementation
    keep = []
    indices = np.argsort(boxes[:, 4])[::-1]  # Sort by confidence
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        # Calculate intersection
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union = area_current + area_others - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        # Keep boxes with IoU below threshold
        indices = indices[1:][iou <= iou_threshold]
    
    return [filtered[i] for i in keep]

def visualize_detections(image, detections, title="YOLOv8 C Model Detections"):
    """Visualize detections on image"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(detections)))
    
    for i, det in enumerate(detections):
        # Convert center coordinates to corner coordinates
        x1 = det['x'] - det['w']/2
        y1 = det['y'] - det['h']/2
        
        # Create rectangle
        rect = Rectangle((x1, y1), det['w'], det['h'], 
                        linewidth=2, edgecolor=colors[i], 
                        facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add label
        label = f"{det['class_name']}: {det['confidence']:.2f}"
        ax.text(x1, y1-5, label, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7),
                color='white', fontweight='bold')
    
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Flip y-axis
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_detection_summary(detections, original_shape, model_shape=(640, 640)):
    """Create a summary of detections"""
    print(f"\n=== Detection Summary ===")
    print(f"Original image size: {original_shape[1]}x{original_shape[0]}")
    print(f"Model input size: {model_shape[0]}x{model_shape[1]}")
    print(f"Total detections: {len(detections)}")
    
    if detections:
        print(f"\nDetection details:")
        for i, det in enumerate(detections):
            print(f"  {i+1}. {det['class_name']}: conf={det['confidence']:.3f}, "
                  f"box=({det['x']:.1f}, {det['y']:.1f}, {det['w']:.1f}, {det['h']:.1f})")
    else:
        print("No detections found above confidence threshold")

def main():
    print("=== YOLOv8 C Model Detection Visualizer ===\n")
    
    # Load image
    print("Loading image...")
    original_img, resized_img, original_shape = load_and_resize_image()
    
    if resized_img is None:
        print("Could not load image. Exiting.")
        return
    
    print(f"Loaded image: {original_shape[1]}x{original_shape[0]} -> 640x640")
    
    # Parse detections
    print("Parsing detections...")
    detections = parse_yolov8_output()
    
    # Apply NMS to filter overlapping detections
    print(f"Found {len(detections)} raw detections")
    filtered_detections = apply_nms(detections, conf_threshold=0.1, iou_threshold=0.5)
    print(f"After NMS: {len(filtered_detections)} detections")
    
    # Create summary
    create_detection_summary(filtered_detections, original_shape)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Plot on resized image (model input size)
    fig1 = visualize_detections(resized_img, filtered_detections, 
                               "YOLOv8 C Model Detections (640x640)")
    
    # Scale detections back to original image size if available
    if original_img is not None:
        scale_x = original_shape[1] / 640
        scale_y = original_shape[0] / 640
        
        scaled_detections = []
        for det in filtered_detections:
            scaled_det = det.copy()
            scaled_det['x'] *= scale_x
            scaled_det['y'] *= scale_y
            scaled_det['w'] *= scale_x
            scaled_det['h'] *= scale_y
            scaled_detections.append(scaled_det)
        
        fig2 = visualize_detections(original_img, scaled_detections, 
                                   f"YOLOv8 C Model Detections (Original {original_shape[1]}x{original_shape[0]})")
    
    # Save plots
    fig1.savefig('yolov8_detections_640x640.png', dpi=150, bbox_inches='tight')
    print("Saved: yolov8_detections_640x640.png")
    
    if original_img is not None:
        fig2.savefig('yolov8_detections_original.png', dpi=150, bbox_inches='tight')
        print("Saved: yolov8_detections_original.png")
    
    # Show plots
    plt.show()
    
    print("\n=== Visualization Complete ===")

if __name__ == "__main__":
    main()