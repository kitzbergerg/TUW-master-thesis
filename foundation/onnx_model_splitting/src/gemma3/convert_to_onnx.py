import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnx
import onnxruntime
import tempfile


# from huggingface_hub import login
# login()

class SimpleModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        # Call the model but ignore cache-related functionality
        outputs = self.model(input_ids, use_cache=False)
        return outputs.logits


def convert_to_onnx(model_id='google/gemma-3-1b-it', output_path='model/gemma3/gemma-3-1b-it.onnx'):
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation='eager')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dummy_input = tokenizer.encode("Hello, world!", return_tensors="pt")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temppath = tmpdir + '/model.onnx'
            torch.onnx.expoptimumort(
                SimpleModelWrapper(model),
                dummy_input,
                temppath,
                opset_version=18,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'logits': {0: 'batch_size', 1: 'sequence'}
                },
                external_data=True
            )

            onnx.save_model(onnx.load(temppath), output_path, save_as_external_data=True, all_tensors_to_one_file=True)

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
