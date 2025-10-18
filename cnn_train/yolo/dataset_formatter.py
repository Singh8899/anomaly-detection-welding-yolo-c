"""
Dataset Formatter Script
Converts XML annotations to YOLO format and organizes dataset structure.

Creates:
dataset/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/

Class mapping:
- weld (both good_weld and bad_weld) → 0
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

def parse_xml_annotation(xml_path):
    """
    Parse XML annotation file and extract bounding box information.
    
    Args:
        xml_path (str): Path to XML annotation file
        
    Returns:
        tuple: (image_width, image_height, objects_list)
               objects_list contains (class_id, x_center, y_center, width, height)
    """
    try:            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        if size is None:
            print(f"Error: No size element found in {xml_path}")
            return None, None, []
            
        width_elem = size.find('width')
        height_elem = size.find('height')
        
        if width_elem is None or height_elem is None:
            print(f"Error: Width or height not found in {xml_path}")
            return None, None, []
            
        img_width = int(width_elem.text)
        img_height = int(height_elem.text)
        
        # Class mapping - single class for all welds
        class_mapping = {
            'bad_weld': 0,
            'good_weld': 0  # Both classes map to same ID for single class detection
        }
        
        objects = []
        
        # Process each object
        for obj in root.findall('object'):
            # Find class name in multiple possible tags
            class_name = None
            
            # Try different possible tag names for class
            for tag_name in ['name', 'class', 'label', 'n']:
                elem = obj.find(tag_name)
                if elem is not None and elem.text is not None:
                    class_name = elem.text.strip()
                    break
            
            if not class_name:
                print(f"Warning: No class name found for object in {xml_path}")
                continue
                
            if class_name not in class_mapping:
                print(f"Warning: Unknown class '{class_name}' in {xml_path}")
                continue
                
            class_id = class_mapping[class_name]
            
            # Get bounding box coordinates
            bbox = obj.find('bndbox')
            if bbox is None:
                print(f"Warning: No bounding box found for object in {xml_path}")
                continue
                
            xmin_elem = bbox.find('xmin')
            xmax_elem = bbox.find('xmax')
            ymin_elem = bbox.find('ymin')
            ymax_elem = bbox.find('ymax')
            
            if None in [xmin_elem, xmax_elem, ymin_elem, ymax_elem]:
                print(f"Warning: Incomplete bounding box coordinates in {xml_path}")
                continue
                
            xmin = int(xmin_elem.text)
            xmax = int(xmax_elem.text)
            ymin = int(ymin_elem.text)
            ymax = int(ymax_elem.text)
            
            # Convert to YOLO format (normalized coordinates)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            objects.append((class_id, x_center, y_center, width, height))
        
        return img_width, img_height, objects
        
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_path}: {e}")
        return None, None, []
    except Exception as e:
        print(f"Unexpected error processing {xml_path}: {e}")
        return None, None, []

def create_yolo_annotation(objects, output_path):
    """
    Create YOLO format annotation file.
    
    Args:
        objects (list): List of (class_id, x_center, y_center, width, height)
        output_path (str): Path to output annotation file
    """
    with open(output_path, 'w') as f:
        for class_id, x_center, y_center, width, height in objects:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def process_dataset_split(source_dir, target_images_dir, target_labels_dir, split_name):
    """
    Process a dataset split (train or val).
    
    Args:
        source_dir (str): Source directory containing images and XML files
        target_images_dir (str): Target directory for images
        target_labels_dir (str): Target directory for label files
        split_name (str): Name of the split for logging
    """
    print(f"\nProcessing {split_name} split...")
    
    # Get all image files
    source_path = Path(source_dir)
    image_files = list(source_path.glob("*.jpeg")) + list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    
    processed_count = 0
    skipped_count = 0
    
    for image_file in image_files:
        # Find corresponding XML file
        xml_file = image_file.with_suffix('.xml')
        
        if not xml_file.exists():
            print(f"Warning: No XML annotation found for {image_file.name}")
            skipped_count += 1
            continue
        
        try:
            # Parse XML annotation
            img_width, img_height, objects = parse_xml_annotation(str(xml_file))
            
            if img_width is None or img_height is None or not objects:
                if img_width is None:
                    print(f"Warning: Could not get image dimensions from {xml_file.name}")
                else:
                    print(f"Warning: No valid objects found in {xml_file.name}")
                skipped_count += 1
                continue
            
            # Copy image to target directory
            target_image_path = os.path.join(target_images_dir, image_file.name)
            shutil.copy2(str(image_file), target_image_path)
            
            # Create YOLO annotation file
            label_filename = image_file.stem + '.txt'
            target_label_path = os.path.join(target_labels_dir, label_filename)
            create_yolo_annotation(objects, target_label_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
            skipped_count += 1
            continue
    
    print(f"{split_name} split: {processed_count} files processed, {skipped_count} files skipped")

def main():
    """Main function to create the dataset structure and convert annotations."""
    
    # Define paths
    original_dataset_path = Path(os.getenv("DATASET_DIR"))
    source_train_dir = original_dataset_path / "train"
    source_val_dir = original_dataset_path/ "val"
    
    processed_dataset_dir = Path(os.getenv("YOLO_DATASET_DIR"))
    train_images_dir = processed_dataset_dir / "train" / "images"
    train_labels_dir = processed_dataset_dir / "train" / "labels"
    val_images_dir = processed_dataset_dir / "val" / "images"
    val_labels_dir = processed_dataset_dir / "val" / "labels"
    
    # Check if source directories exist
    if not source_train_dir.exists():
        print(f"Error: Source train directory not found: {source_train_dir}")
        return
    
    if not source_val_dir.exists():
        print(f"Error: Source validation directory not found: {source_val_dir}")
        return
    
    # Create dataset directory structure
    print("Creating dataset directory structure...")
    
    # Remove existing dataset directory if it exists
    if processed_dataset_dir.exists():
        print("Removing existing dataset directory...")
        shutil.rmtree(processed_dataset_dir)
    
    # Create new directory structure
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    val_images_dir.mkdir(parents=True, exist_ok=True)
    val_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created:")
    print(f"  {processed_dataset_dir}")
    print(f"  ├── train/")
    print(f"  │   ├── images/")
    print(f"  │   └── labels/")
    print(f"  └── val/")
    print(f"      ├── images/")
    print(f"      └── labels/")
    
    # Process train split
    process_dataset_split(
        str(source_train_dir),
        str(train_images_dir),
        str(train_labels_dir),
        "Train"
    )
    
    # Process validation split
    process_dataset_split(
        str(source_val_dir),
        str(val_images_dir),
        str(val_labels_dir),
        "Validation"
    )
    
    # Print summary
    print(f"\n{'='*50}")
    print("Dataset conversion completed!")
    print(f"{'='*50}")
    
    # Count final files
    train_images_count = len(list(train_images_dir.glob("*")))
    train_labels_count = len(list(train_labels_dir.glob("*.txt")))
    val_images_count = len(list(val_images_dir.glob("*")))
    val_labels_count = len(list(val_labels_dir.glob("*.txt")))
    
    print(f"Train set: {train_images_count} images")
    print(f"Validation set: {val_images_count} images")
    print(f"Total: {train_images_count + val_images_count} images")
    
    print(f"\nDataset saved to: {processed_dataset_dir}")
    print("\nClass mapping used:")
    print("  0: bad_weld")
    print("  0: good_weld")

if __name__ == "__main__":
    main()
