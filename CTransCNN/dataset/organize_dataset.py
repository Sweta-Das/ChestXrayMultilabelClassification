import pandas as pd
import os
import shutil

csv_filename = 'Data_Entry_2017.csv'
source_images_folder = 'images-224'

# Get the current directory, which is the base path for the dataset
dataset_base_path = os.getcwd()
csv_path = os.path.join(dataset_base_path, csv_filename)
source_images_path = os.path.join(dataset_base_path, source_images_folder)
print(f"Reading dataset manifest from: {csv_path}")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"FATAL: The file {csv_path} was not found. Please ensure this script is inside your dataset folder.")
    exit()

print("Successfully loaded CSV. Starting to organize images...")
processed_count = 0

# Iterate over each row in the CSV file
for index, row in df.iterrows():
    image_filename = row['Image Index']
    labels = row['Finding Labels'].split('|')
    
    source_image_full_path = os.path.join(source_images_path, image_filename)
    
    # Check if the source image actually exists
    if not os.path.exists(source_image_full_path):
        print(f"Warning: Image file not found, skipping -> {source_image_full_path}")
        continue
        
    processed_count += 1
    # For each label an image has ('No Finding', 'Cardiomegaly', etc.)
    for label in labels:
        # Create a directory for the label if it doesn't exist
        # The new directory will be at the same level as 'images-224'
        label_dir = os.path.join(dataset_base_path, label)
        os.makedirs(label_dir, exist_ok=True)
        
        destination_image_full_path = os.path.join(label_dir, image_filename)
        
        # Copy the image file to the label's directory
        shutil.copy2(source_image_full_path, destination_image_full_path)

    # Progress indicator
    if (index + 1) % 5000 == 0:
        print(f"Processed {index + 1} of {len(df)} records...")

print("\n--- Image Organization Complete! ---")
print(f"Successfully organized {processed_count} images into label-specific folders.")