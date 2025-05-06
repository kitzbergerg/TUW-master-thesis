from optimum.exporters.onnx import main_export


# Or use optimum-cli:
# > optimum-cli export onnx --model gpt2 --opset 18 --optimize O2 model/gpt2/optimum/

def convert_to_onnx():
    main_export(
        'gpt2',
        opset=18,
        optimize='O2',
        output='model/gpt2/optimum',
    )


if __name__ == "__main__":
    convert_to_onnx()
