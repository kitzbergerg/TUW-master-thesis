import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnx
import onnxruntime
from transformers.cache_utils import DynamicCache


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, legacy_cache):
        assert len(legacy_cache) == self.model.config.num_hidden_layers
        assert len(legacy_cache[0]) == 2
        past = DynamicCache.from_legacy_cache(legacy_cache)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
            return_dict=True
        )

        cache: DynamicCache = outputs.past_key_values
        return (outputs.logits,) + cache.to_legacy_cache()


def create_legacy_cache(model, batch_size=1, seq_len=1, past_seq_len=5, device="cpu"):
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads

    def empty_kv():
        return torch.zeros(batch_size, num_heads, past_seq_len, head_dim, device=device)

    num_layers = model.config.num_hidden_layers
    legacy_cache = []
    for _ in range(num_layers):
        legacy_cache.append((empty_kv(), empty_kv()))
    return tuple(legacy_cache)


def prepare_inputs(model, tokenizer, text="Hello"):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    legacy_cache = create_legacy_cache(model, batch_size=batch_size, seq_len=seq_len, device=device)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'legacy_cache': legacy_cache
    }


def convert_to_onnx(output_path, wrapped_model, inputs):
    model = wrapped_model.model
    num_layers = model.config.num_hidden_layers
    input_names = ['input_ids', 'attention_mask', 'position_ids'] + [
        f'past_key_values.{i}.{t}' for i in range(num_layers) for t in ['key', 'value']
    ]
    output_names = ['logits'] + [
        f'present.{i}.{t}' for i in range(num_layers) for t in ['key', 'value']
    ]
    input_tuple = (
        inputs['input_ids'],
        inputs['attention_mask'],
        inputs['position_ids'],
        inputs['legacy_cache']
    )

    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'past_sequence_length + 1'},
        'position_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'},

        **{name: {0: 'batch_size', 1: f'{num_heads}', 2: 'past_sequence_length', 3: f'{head_dim}'}
           for name in input_names if
           'past_key_values' in name},
        **{name: {0: 'batch_size', 1: f'{num_heads}', 2: 'past_sequence_length + 1', 3: f'{head_dim}'}
           for name in output_names if 'present' in name}
    }

    try:
        torch.onnx.export(
            wrapped_model,
            input_tuple,
            output_path,
            opset_version=18,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            export_params=True
        )
        onnx.checker.check_model(output_path)
        print(f"Model successfully converted and saved to {output_path}")

    except Exception as e:
        print(f"Error converting model to ONNX: {e}")
        import traceback
        traceback.print_exc()


def verify_onnx_model(onnx_path, pytorch_model, inputs):
    session = onnxruntime.InferenceSession(onnx_path)
    num_layers = pytorch_model.model.config.num_hidden_layers
    ort_inputs = {
        'input_ids': inputs['input_ids'].numpy(),
        'attention_mask': inputs['attention_mask'].numpy(),
        'position_ids': inputs['position_ids'].numpy(),
        **{f'past_key_values.{i}.key': inputs['legacy_cache'][i][0].numpy() for i in range(num_layers)},
        **{f'past_key_values.{i}.value': inputs['legacy_cache'][i][1].numpy() for i in range(num_layers)},
    }

    ort_outputs = session.run(None, ort_inputs)

    with torch.no_grad():
        pt_outputs = pytorch_model(
            inputs['input_ids'],
            inputs['attention_mask'],
            inputs['position_ids'],
            inputs['legacy_cache']
        )

    def compare(a, b, name):
        if torch.allclose(a, b, atol=1e-4):
            print(f"{name} verification successful!")
        else:
            print(f"WARNING: {name} differs. Max diff: {(a - b).abs().max()}")

    compare(torch.tensor(ort_outputs[0]), pt_outputs[0], "Logits")

    for i in range(1, len(pt_outputs)):
        compare(torch.tensor(ort_outputs[2 * i]), pt_outputs[i][0], f"Layer {i} key")
        compare(torch.tensor(ort_outputs[2 * i + 1]), pt_outputs[i][1], f"Layer {i} value")


model_name = 'microsoft/phi-2'
output_path = 'model/phi/torch/model.onnx'

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(model_name)
    wrapped_model = ModelWrapper(model)
    inputs = prepare_inputs(model, AutoTokenizer.from_pretrained(model_name))

    convert_to_onnx(output_path, wrapped_model, inputs)
    verify_onnx_model(output_path, wrapped_model, inputs)
