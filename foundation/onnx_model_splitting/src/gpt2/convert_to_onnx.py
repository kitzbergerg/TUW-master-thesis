import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import onnx
import onnxruntime


def convert_to_onnx(model_name='gpt2', output_path='model/gpt2/gpt2.onnx'):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    dummy_input = tokenizer.encode("Hello, world!", return_tensors="pt")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=18,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size', 1: 'sequence'}
            }
        )

        onnx.checker.check_model(output_path)
        print(f"Model successfully converted and saved to {output_path}")
        verify_onnx_model(output_path, model, dummy_input)
    except Exception as e:
        print(f"Error converting model to ONNX: {e}")


def verify_onnx_model(onnx_path, pytorch_model, input_ids):
    session = onnxruntime.InferenceSession(onnx_path)

    ort_inputs = {session.get_inputs()[0].name: input_ids.numpy()}

    ort_outputs = session.run(None, ort_inputs)

    with torch.no_grad():
        pt_outputs = pytorch_model(input_ids)

    onnx_output = torch.tensor(ort_outputs[0])
    pt_output = pt_outputs.logits

    if torch.allclose(onnx_output, pt_output, atol=1e-5):
        print("ONNX model verification successful!")
    else:
        print("WARNING: ONNX model outputs differ from PyTorch model")


if __name__ == "__main__":
    convert_to_onnx()
