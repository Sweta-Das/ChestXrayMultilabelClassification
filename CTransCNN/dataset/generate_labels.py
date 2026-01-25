# generate_labels.py (Compatible with Python 2.7 and 3+)
import pandas as pd
import os

# --- Configuration ---
csv_filename = 'Data_Entry_2017.csv'
train_val_list_filename = 'train_val_list_NIH.txt'
test_list_filename = 'test_list_NIH.txt'

# These are the names for the new, correct output files
output_train_filename = 'chest14_train_labels.txt'
output_val_filename = 'chest14_val_labels.txt'
output_test_filename = 'chest14_test_labels.txt'
output_classes_filename = 'chest14_classes.txt'

# 10% of the train_val list will be used for validation
validation_split_ratio = 0.1
# --- End of Configuration ---

print("Starting label generation...")

# 1. Get the 14 official classes from the dataset CSV
df = pd.read_csv(csv_filename)
all_labels = set()
for labels in df['Finding Labels']:
    for label in labels.split('|'):
        all_labels.add(label)

if 'No Finding' in all_labels:
    all_labels.remove('No Finding')

classes = sorted(list(all_labels))
print('Found {} classes.'.format(len(classes)))

# Write the classes to a file
with open(output_classes_filename, 'w') as f:
    for cls in classes:
        f.write('{}\n'.format(cls))
print("Successfully created '{}'".format(output_classes_filename))

# 2. Create a fast lookup map from image index to its labels
image_to_labels_map = {row['Image Index']: row['Finding Labels'].split('|') for index, row in df.iterrows()}

# 3. Define the function to create a label file
def create_label_file(image_list, output_filename):
    print("Processing '{}'...".format(output_filename))
    with open(output_filename, 'w') as f:
        for image_file in image_list:
            image_file = image_file.strip()
            if not image_file:
                continue

            labels = image_to_labels_map.get(image_file, [])
            # The first label determines the folder path
            primary_label = labels[0] if labels else "Unknown"
            image_path = os.path.join(primary_label, image_file)

            # Create the one-hot encoded vector (e.g., 0,1,0,0,...)
            binary_vector = [1 if cls in labels else 0 for cls in classes]
            vector_str = ','.join(map(str, binary_vector))

            f.write('{}\t{}\n'.format(image_path, vector_str))
    print("Successfully created '{}'".format(output_filename))

# 4. Process the train, validation, and test lists
with open(test_list_filename, 'r') as f:
    test_images = f.readlines()
create_label_file(test_images, output_test_filename)

with open(train_val_list_filename, 'r') as f:
    train_val_images = f.readlines()

# Split the main list into separate training and validation sets
split_index = int(len(train_val_images) * (1 - validation_split_ratio))
train_images = train_val_images[:split_index]
val_images = train_val_images[split_index:]

create_label_file(train_images, output_train_filename)
create_label_file(val_images, output_val_filename)

print("\n--- Label Generation Complete! ---")