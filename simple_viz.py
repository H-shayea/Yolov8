#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# Load PPM image manually (since we don't have opencv)
def load_ppm_image(filepath):
    with open(filepath, 'rb') as f:
        magic = f.readline().decode().strip()
        if magic != 'P6':
            raise ValueError("Only P6 PPM format supported")
        
        # Skip comments
        line = f.readline().decode().strip()
        while line.startswith('#'):
            line = f.readline().decode().strip()
        
        width, height = map(int, line.split())
        maxval = int(f.readline().decode().strip())
        
        # Read image data
        data = f.read(width * height * 3)
        image = np.frombuffer(data, dtype=np.uint8)
        image = image.reshape((height, width, 3))
    
    return image

# Parse detections from our C model output
def load_detections(filepath):
    detections = []
    class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck']
    
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('Det'):
                # Extract from "Det 0: 25.9 421.8 51.8 188.8 0.313 0"
                parts = line.split(':')[1].strip().split()
                if len(parts) == 6:
                    x, y, w, h, conf, cls = map(float, parts)
                    class_name = class_names[int(cls)] if int(cls) < len(class_names) else f'class_{int(cls)}'
                    detections.append((x, y, w, h, conf, class_name))
    return detections

# Main visualization
def visualize():
    # Load image
    image = load_ppm_image('images/street.ppm')
    print(f"Image shape: {image.shape}")
    
    # Load detections
    detections = load_detections('c_detections.txt')
    print(f"Found {len(detections)} detections")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Street Image')
    ax1.axis('off')
    
    # Image with detections
    ax2.imshow(image)
    ax2.set_title(f'YOLOv8 C Model Detections ({len(detections)} boxes)')
    
    # Draw bounding boxes
    img_h, img_w = image.shape[:2]
    
    colors = {'person': 'red', 'bus': 'blue', 'car': 'green', 'bicycle': 'orange'}
    
    for i, (x, y, w, h, conf, class_name) in enumerate(detections):
        # YOLOv8 outputs are in 640x640 normalized format
        # Convert center coordinates to corner coordinates
        x_center = x * img_w / 640
        y_center = y * img_h / 640  
        box_w = w * img_w / 640
        box_h = h * img_h / 640
        
        # Convert center format to corner format
        x_corner = x_center - box_w / 2
        y_corner = y_center - box_h / 2
        
        # Choose color based on class
        color = colors.get(class_name, 'red')
        
        # Draw rectangle
        rect = plt.Rectangle((x_corner, y_corner), box_w, box_h,
                           linewidth=2, edgecolor=color, facecolor='none')
        ax2.add_patch(rect)
        
        # Add label with class and confidence
        label = f'{class_name}: {conf:.2f}'
        ax2.text(x_corner, y_corner-10, label, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8),
                fontsize=8, color='black')
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('detections_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Saved visualization as 'detections_visualization.png'")
    
    # Print detection details  
    print("\nDetection summary:")
    class_counts = {}
    for i, (x, y, w, h, conf, class_name) in enumerate(detections):
        print(f"  {class_name}: confidence={conf:.2f}, center=({x:.0f},{y:.0f}), size=({w:.0f}x{h:.0f})")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"\nClass counts: {class_counts}")

if __name__ == "__main__":
    visualize()