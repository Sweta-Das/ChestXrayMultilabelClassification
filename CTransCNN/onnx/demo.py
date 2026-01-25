import onnxruntime as ort
import numpy as np
from PIL import Image
import time

def load_class_names_from_txt(txt_path):
    """Loads class names from a text file, one name per line."""
    with open(txt_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def load_image(image_path, size=(224, 224)):
    """Loads and preprocesses the image for your specific CTransCNN model."""
    image = Image.open(image_path)
    image = image.resize(size).convert('RGB')
    image_array = np.array(image).astype(np.float32)

    # --- MODEL NORMALIZATION ---
    # Values are taken directly from config file.
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    image_array = (image_array - mean) / std

    image_array = np.transpose(image_array, [2, 0, 1])  # Change to CHW format (Channels, Height, Width)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension (1, C, H, W)
    return image_array

def predict(image_path, model_path):
    """Loads the ONNX model, runs the prediction, and processes the output."""
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    image = load_image(image_path)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    start_time = time.time()
    # The output from the model is a list containing one array
    scores = session.run([output_name], {input_name: image})[0]
    time_cost = time.time() - start_time
    
    return scores, time_cost

def run_example(image_path, model_path, class_names_path):
    """Executes the model prediction and prints the results."""
    class_names = load_class_names_from_txt(class_names_path)
    scores, time_cost = predict(image_path, model_path)

    # Sigmoid function to convert logits to probabilities between 0 and 1
    probabilities = 1 / (1 + np.exp(-scores[0]))
    
    threshold = 0.5

    print("--- Prediction Results ---")
    print(f"Image: {image_path.split('/')[-1]}")
    print(f"Found the following conditions (threshold > {threshold}):")
    
    found_any = False
    for idx, prob in enumerate(probabilities):
        if prob > threshold:
            print(f"- {class_names[idx]} (Confidence: {prob:.2%})")
            found_any = True
    
    if not found_any:
        print("- No conditions found above the threshold.")

    print("\nInference time: {:.4f} seconds".format(time_cost))

# Model path
model_path = 'onnx/model_epoch_72.onnx' 

# Test file
image_path = 'onnx/1.2.826.0.1.3680043.8.498.41477985046369504069935954327112429508.jpg'

# Class names file.
class_names_path = 'dataset/chest14_classes.txt'

run_example(image_path, model_path, class_names_path)