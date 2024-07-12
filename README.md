

# YOLOv8seg Annotation File Converter

This repository contains a Python script that converts annotation files generated by CVAT (using the SAM model) into the required format for training a YOLOv8seg model. The script creates `.txt` files containing bounding box coordinates and class labels.

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/nicholas.m.aplin/cvat-segmention-mask-to-yolo.git
   cd cvat-segmention-mask-to-yolo
   ```

2. **Install Dependencies:**
   Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Script:**
   Execute the `convert_annotations.py` script, passing the path to your CVAT annotation file as an argument:
   ```bash
   python cvat_to_yolo.py --input json_file_name.json, /path/to/folder
   ```

4. **Output:**
   The script will generate `.txt` files in the YOLO format (one per image) containing bounding box coordinates and class labels.

## Example
Check out the `example.py` file in this repository for a usage example.


