# inference.py

import onnxruntime as ort
from utils.preprocess import preprocess_image

MODEL_PATH = "models/ctranscnn_1.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def predict(img_file):
    """
    Returns:
        probs: list of length 14 (multi-label probabilities)
    """
    x = preprocess_image(img_file)

    outputs = session.run(
        [output_name],
        {input_name: x}
    )

    probs = outputs[0][0].tolist()
    return probs
