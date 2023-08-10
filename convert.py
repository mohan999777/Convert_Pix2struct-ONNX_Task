

# Load the Pix2Struct base model and tokenizer from the Hugging Face library

from transformers import Pix2StructForDocumentQuestionAnswering, Pix2StructTokenizer

model_name = "google/pix2struct-docvqa-base"
tokenizer = Pix2StructTokenizer.from_pretrained(model_name)
model = Pix2StructForDocumentQuestionAnswering.from_pretrained(model_name)

# Prepare Example Input: Create an example input that matches the model's input specifications

image = "home/desktop/cristiano_ronaldo.jpg"
text = "Ronaldo lifting a UCL trophy in 2017"
inputs = tokenizer(image, text, return_tensors="pt")

# Convert the model to ONNX using TensorFlow and the ONNX-TensorFlow bridge

import tensorflow as tf
import tf2onnx

# Convert the PyTorch model to a TensorFlow model

dummy_input = {"input_ids": inputs["input_ids"].numpy(), "attention_mask": inputs["attention_mask"].numpy()}
onnx_model_path = "pix2struct.onnx"

# Convert the model to ONNX using the ONNX-TensorFlow bridge

with tf.Graph().as_default():
    tf_rep = tf2onnx.convert.from_keras(model, input_signature=dummy_input)
    tf_rep.export_graph(onnx_model_path)

print(f"Model converted to ONNX and saved at {onnx_model_path}")


