from pathlib import Path

from transformers.onnx import export
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt2 import GPT2OnnxConfig


def convert_to_onnx():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    onnx_config = GPT2OnnxConfig(model.config, use_past=True)

    export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=18,
        output=Path("model/gpt2/model_transformers.onnx"),
    )


if __name__ == "__main__":
    convert_to_onnx()
