import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import onnx
import onnxruntime


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, *past_key_values):
        past = [(past_key_values[i], past_key_values[i + 1]) for i in range(0, len(past_key_values), 2)]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past if past and past[0][0].size(2) > 0 else None,
            use_cache=True,
            return_dict=True
        )

        ordered_outputs = []
        for layer_idx in range(len(outputs.past_key_values)):
            ordered_outputs.append(outputs.past_key_values[layer_idx][0])
            ordered_outputs.append(outputs.past_key_values[layer_idx][1])
        return (outputs.logits,) + tuple(ordered_outputs)


def create_inputs(model, tokenizer, text="Hello"):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    past_key_values_length = 5
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

    head_dim = model.config.n_embd // model.config.n_head
    past = lambda: torch.zeros(batch_size, model.config.n_head, past_key_values_length, head_dim, device=device)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        **{f'past_key_values.{i}.{t}': past() for i in range(model.config.n_layer) for t in ['key', 'value']},
    }


def convert_to_onnx(model_name='gpt2', output_path='model/gpt2/model_torch.onnx'):
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    wrapped_model = ModelWrapper(model)
    inputs = create_inputs(model, tokenizer)

    n_layer = model.config.n_layer
    input_names = (['input_ids', 'attention_mask', 'position_ids'] +
                   [f'past_key_values.{i}.{t}' for i in range(n_layer) for t in ['key', 'value']])
    output_names = (['logits'] +
                    [f'present.{i}.{t}' for i in range(n_layer) for t in ['key', 'value']])

    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'past_sequence_length + 1'},
        'position_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length', 2: 'vocab_size'},

        **{name: {0: 'batch_size', 2: 'past_sequence_length'} for name in input_names if 'past_key_values.' in name},
        **{name: {0: 'batch_size', 2: 'past_sequence_length + 1'} for name in output_names if 'present.' in name}
    }

    try:
        input_tuple = tuple(inputs[name] for name in input_names)
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
        verify_onnx_model(output_path, wrapped_model, inputs)

    except Exception as e:
        print(f"Error converting model to ONNX: {e}")
        import traceback
        traceback.print_exc()


def verify_onnx_model(onnx_path, pytorch_model, inputs):
    session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {i.name: inputs[i.name].numpy() for i in session.get_inputs()}
    ort_outputs = session.run(None, ort_inputs)

    with torch.no_grad():
        pt_input = tuple(inputs[i.name] for i in session.get_inputs())
        pt_outputs = pytorch_model(*pt_input)

    def compare(a, b, name):
        if torch.allclose(a, b, atol=1e-4):
            print(f"{name} verification successful!")
        else:
            print(f"WARNING: {name} differs. Max diff: {(a - b).abs().max()}")

    compare(torch.tensor(ort_outputs[0]), pt_outputs[0], "Logits")

    for i in range(pytorch_model.model.config.n_layer):
        compare(torch.tensor(ort_outputs[2 * i + 1]), pt_outputs[2 * i + 1], f"Layer {i} key")
        compare(torch.tensor(ort_outputs[2 * i + 2]), pt_outputs[2 * i + 2], f"Layer {i} value")


if __name__ == "__main__":
    convert_to_onnx()
