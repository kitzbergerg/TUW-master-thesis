from optimum.exporters.onnx import main_export


def convert_to_onnx():
    main_export(
        'microsoft/phi-2',
        opset=18,
        output='model/phi/optimum',
    )


if __name__ == "__main__":
    convert_to_onnx()
