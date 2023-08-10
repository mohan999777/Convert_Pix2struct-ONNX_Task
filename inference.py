import argparse
import cv2
import numpy as np
import onnxruntime

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Adjust the dimensions if needed
    image = image.astype(np.float32) / 255.0  # normalizing
    image = np.transpose(image, (2, 0, 1))  # Convert to channel-first format
    return image

def main():
    parser = argparse.ArgumentParser(description="Run inference using ONNX model")
    parser.add_argument("-m", "--model", required=True, help="Path to the ONNX model folder")
    parser.add_argument("-i", "--image", required=True, help="Path to the input image")
    args = parser.parse_args()

    # Load the ONNX model
    onnx_model_path = args.model
    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    # Load and preprocess the image
    image_path = args.image
    input_image = load_image(image_path)
    input_name = ort_session.get_inputs()[0].name
    inputs = {input_name: np.expand_dims(input_image, axis=0)}

    # Run inference
    outputs = ort_session.run(None, inputs)

    # Print the inference result
    print("Inference Result:")
    print(outputs)

if __name__ == "__main__":
    main()

