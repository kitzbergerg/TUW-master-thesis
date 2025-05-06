import torch
import tempfile
import onnx
import onnxruntime as rt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import HybridCache


class GemmaWrapper(torch.nn.Module):
    def __init__(self, model, past_cache: HybridCache):
        super().__init__()
        self.model = model
        self.past = past_cache

    def forward(self, input_ids, position_ids, past_keys, past_values):
        self.past.reset()
        for i, (k, v) in enumerate(zip(past_keys, past_values)):
            self.past.update(k, v, i)

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=self.past,
            use_cache=True
        )

        logits = outputs.logits
        new_past = outputs.past_key_values
        return (logits,) + tuple(new_past.key_cache + new_past.value_cache)


def get_inputs(tokenizer, text="Hello, world!"):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    return input_ids, position_ids


def convert_to_onnx(model_id, output_path):
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_ids, position_ids = get_inputs(tokenizer)

    # Warm-up to get cache structure
    outputs = model(input_ids=input_ids, position_ids=position_ids, use_cache=True)
    past: HybridCache = outputs.past_key_values
    n_layers = len(past.key_cache)

    # ONNX export
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = f"{tmpdir}/model.onnx"
        torch.onnx.export(
            GemmaWrapper(model, past),
            (input_ids, position_ids, past.key_cache, past.value_cache),
            temp_path,
            input_names=["input_ids", "position_ids"] +
                        [f"past_key_values.{i}.key" for i in range(n_layers)] +
                        [f"past_key_values.{i}.value" for i in range(n_layers)],
            output_names=["logits"] +
                         [f"present.{i}.key" for i in range(n_layers)] +
                         [f"present.{i}.value" for i in range(n_layers)],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "position_ids": {0: "batch_size", 1: "seq_len"},
                **{f"past_key_values.{i}.key": {0: "batch_size", 2: "past_seq_len"} for i in range(n_layers)},
                **{f"past_key_values.{i}.value": {0: "batch_size", 2: "past_seq_len"} for i in range(n_layers)},
                **{f"present.{i}.key": {0: "batch_size", 2: "total_seq_len"} for i in range(n_layers)},
                **{f"present.{i}.value": {0: "batch_size", 2: "total_seq_len"} for i in range(n_layers)},
            },
            opset_version=18,
            do_constant_folding=True,
            external_data=True
        )

        # Save with external data
        onnx.save_model(
            onnx.load(temp_path),
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location='model.onnx_data'
        )

    onnx.checker.check_model(output_path)
    print(f"Exported ONNX model saved to {output_path}")


def verify_onnx_model(model_id, onnx_path):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    input_ids, position_ids = get_inputs(tokenizer)

    session = rt.InferenceSession(onnx_path)
    n_layers = (len(session.get_inputs()) - 2) // 2

    # Create dummy empty past cache
    dummy_shape = [1, 1, 0, 256]
    ort_inputs = {
        "input_ids": input_ids.numpy(),
        "position_ids": position_ids.numpy(),
        **{f"past_key_values.{i}.key": torch.zeros(dummy_shape).numpy() for i in range(n_layers)},
        **{f"past_key_values.{i}.value": torch.zeros(dummy_shape).numpy() for i in range(n_layers)},
    }

    ort_outputs = session.run(None, ort_inputs)
    pt_logits = model(input_ids).logits

    onnx_logits = torch.tensor(ort_outputs[0])
    if torch.allclose(onnx_logits, pt_logits, atol=1e-5):
        print("ONNX output matches PyTorch output")
    else:
        print("Output mismatch between ONNX and PyTorch")


if __name__ == "__main__":
    model_id = "google/gemma-3-1b-it"
    onnx_output_path = "model/gemma3/gemma-3-1b-it-kv.onnx"
    convert_to_onnx(model_id, onnx_output_path)
    verify_onnx_model(model_id, '../onnxruntime_web_browser_execution/model/gemma3/model_q4.onnx')
    # TODO: Fails with exception.
    #  It seems like the model doesn't support cache of length 0, so there is some error in the export
    verify_onnx_model(model_id, onnx_output_path)
