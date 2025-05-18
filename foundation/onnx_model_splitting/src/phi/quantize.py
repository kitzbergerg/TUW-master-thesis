from onnxruntime.quantization import quantize_dynamic

model_fp32_path = "model/phi/optimum/model.onnx"
model_quant_path = "model/phi/quantized/model.onnx"


def quantize():
    quantize_dynamic(
        model_input=model_fp32_path,
        model_output=model_quant_path,
        use_external_data_format=True
    )


if __name__ == '__main__':
    quantize()
