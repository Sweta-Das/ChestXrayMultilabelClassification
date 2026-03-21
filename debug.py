import onnxruntime as ort
import numpy as np

MODEL_PATH = "models/ctranscnn_1.onnx"  # change if needed


def main():
    print("=" * 60)
    print("ONNX MODEL STRUCTURE DEBUG")
    print("=" * 60)

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    # -----------------------------
    # 1. Inputs
    # -----------------------------
    print("\n[1] MODEL INPUTS")
    for i, inp in enumerate(session.get_inputs()):
        print(f"  Input {i}:")
        print(f"    name : {inp.name}")
        print(f"    shape: {inp.shape}")
        print(f"    type : {inp.type}")

    # -----------------------------
    # 2. Outputs
    # -----------------------------
    print("\n[2] MODEL OUTPUTS")
    for i, out in enumerate(session.get_outputs()):
        print(f"  Output {i}:")
        print(f"    name : {out.name}")
        print(f"    shape: {out.shape}")
        print(f"    type : {out.type}")

    # -----------------------------
    # 3. Dummy inference
    # -----------------------------
    print("\n[3] DUMMY INFERENCE OUTPUT ANALYSIS")

    # Create dummy input matching input shape
    input_shape = session.get_inputs()[0].shape
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    outputs = session.run(
        None,
        {session.get_inputs()[0].name: dummy_input}
    )

    for i, out in enumerate(outputs):
        out = np.array(out)
        print(f"\n  Output {i}:")
        print(f"    actual shape : {out.shape}")
        print(f"    min / max    : {out.min():.4f} / {out.max():.4f}")
        print(f"    mean         : {out.mean():.4f}")

        if out.ndim == 2:
            print(f"    vector length: {out.shape[1]}")

    # -----------------------------
    # 4. Classification head check
    # -----------------------------
    print("\n[4] CLASSIFICATION HEAD CHECK")

    found_classifier = False
    for i, out in enumerate(outputs):
        if out.ndim == 2 and out.shape[1] == 14:
            found_classifier = True
            print(f"  ✅ Output {i} LOOKS LIKE CLASSIFIER OUTPUT (14 classes)")

            # Check if already sigmoid-ed
            if out.min() >= 0.0 and out.max() <= 1.0:
                print("     → Values in [0,1]: sigmoid already applied")
            else:
                print("     → Values outside [0,1]: raw logits")

    if not found_classifier:
        print("  ❌ NO 14-DIMENSION OUTPUT FOUND")
        print("     → Model is ENCODER-ONLY or classifier not exposed")

    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
