import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(img_file):
    img = Image.open(img_file).convert("RGB")
    img = img.resize((224, 224))

    x = np.array(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD

    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)

    return x
